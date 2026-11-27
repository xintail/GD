#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
广东省火灾像元二分类：多模型、按日期交叉验证、F1 阈值搜索。

输入
----
每日特征：
    ./data/processed/features_1km/feature_YYYYMMDD.npy
    shape = (72, H, W)

每日 FIRMS confidence：
    ./data/processed/firms_conf_1km/firms_conf_YYYYMMDD.npy
    shape = (H, W)

广东掩膜：
    ./data/processed/guangdong_mask_1km.npy

通道元数据：
    ./data/processed/feature_channel_metadata.json

标签
----
FIRMS confidence > LABEL_CONFIDENCE_THRESHOLD 记为火灾 1，
否则为非火灾 0。默认阈值是 0，即只要存在 FIRMS 火点就记为正类。

交叉验证
--------
只选择“至少有一个火灾像元”的日期作为验证日。
每折留出其中 1 天作为验证集，其余全部日期作为训练集。
预处理器只在训练日期拟合。

将所有折的验证概率合并成 OOF 概率，
再选择使整体 F1 最大的全局分类阈值。

气压层
------
用户要求 1000、925、800 hPa。
若元数据中没有某一层，默认选择最接近的可用层。
在当前 13 层配置中，800 hPa 通常会映射到 850 hPa，
映射会写入每个 checkpoint 的 selected_channels.json。

输出
----
checkpoint/<model_name>/
    model.joblib
    threshold.json
    metrics.json
    fold_metrics.csv
    threshold_search.csv
    selected_channels.json
    training_config.json
    training_log.txt
    oof_predictions.npz
    oof_day_slices.json
    roc_curve.png
    pr_curve.png
    threshold_f1_curve.png
    confusion_matrix.png
"""

from __future__ import annotations

import csv
import gc
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from model.wildfire_model_zoo import MODEL_NAMES, build_model


# =============================================================================
# 0. 配置
# =============================================================================
START_DATE = "2025-12-01"
END_DATE = "2025-12-31"

FEATURE_DIR = Path("./data/processed/features_1km")
FEATURE_PATTERN = "feature_{date}.npy"

LABEL_DIR = Path("./data/processed/firms_conf_1km")
LABEL_PATTERN = "firms_conf_{date}.npy"

MASK_PATH = Path("./data/processed/guangdong_mask_1km.npy")
CHANNEL_METADATA_PATH = Path(
    "./data/processed/feature_channel_metadata.json"
)

CHECKPOINT_ROOT = Path("./checkpoint")

# 同时训练多个模型
ENABLED_MODELS = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
]

# 标签二值化：confidence > 0 为火灾
LABEL_CONFIDENCE_THRESHOLD = 0.0

# surface 4 个变量全部保留；upper 只保留以下三层。
REQUESTED_PRESSURE_LEVELS = [1000, 925, 800]
ALLOW_NEAREST_PRESSURE_LEVEL = True

# 训练负样本抽样。
# 正样本全部保留，负样本只从广东掩膜内抽取。
NEGATIVE_TO_POSITIVE_RATIO = 20
MIN_NEGATIVES_PER_FIRE_DAY = 5000
MAX_NEGATIVES_PER_DAY = 100000
NO_FIRE_DAY_NEGATIVE_SAMPLES = 10000

RANDOM_STATE = 42
PREDICT_BATCH_SIZE = 100000

# 是否覆盖已有 checkpoint
OVERWRITE_CHECKPOINT = True

# 是否保存 OOF 像元概率
SAVE_OOF_PREDICTIONS = True

# 绘图分辨率
FIGURE_DPI = 250


# =============================================================================
# 1. 基础工具
# =============================================================================
def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d")


def date_range(start: str, end: str) -> Iterable[str]:
    current = parse_date(start)
    end_dt = parse_date(end)

    if end_dt < current:
        raise ValueError("END_DATE 不能早于 START_DATE。")

    while current <= end_dt:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def append_log(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def validate_model_names(names: Sequence[str]) -> None:
    invalid = [name for name in names if name not in MODEL_NAMES]

    if invalid:
        raise ValueError(
            f"未知模型：{invalid}；可选模型：{MODEL_NAMES}"
        )


# =============================================================================
# 2. 通道选择
# =============================================================================
def resolve_pressure_levels(
    channel_order: Sequence[Dict[str, object]],
) -> Tuple[List[int], Dict[int, int]]:
    available_levels = sorted(
        {
            int(item["pressure_level_hpa"])
            for item in channel_order
            if item.get("source") == "weather_upper"
            and item.get("pressure_level_hpa") is not None
        },
        reverse=True,
    )

    if not available_levels:
        raise ValueError("元数据中没有 weather_upper 气压层。")

    selected_actual: List[int] = []
    mapping: Dict[int, int] = {}

    for requested in REQUESTED_PRESSURE_LEVELS:
        if requested in available_levels:
            actual = requested
        elif ALLOW_NEAREST_PRESSURE_LEVEL:
            actual = min(
                available_levels,
                key=lambda level: abs(level - requested),
            )
        else:
            raise ValueError(
                f"请求 {requested} hPa，但可用层为 {available_levels}"
            )

        if actual not in selected_actual:
            selected_actual.append(actual)

        mapping[int(requested)] = int(actual)

    return selected_actual, mapping


def select_feature_channels(
    metadata: Dict[str, object],
) -> Tuple[List[int], List[Dict[str, object]], Dict[int, int]]:
    channel_order = metadata["channel_order"]
    selected_levels, level_mapping = resolve_pressure_levels(
        channel_order
    )

    selected_items: List[Dict[str, object]] = []

    for item in channel_order:
        source = item.get("source")

        if source == "weather_surface":
            selected_items.append(dict(item))

        elif source == "weather_upper":
            level = item.get("pressure_level_hpa")
            if level is not None and int(level) in selected_levels:
                selected_items.append(dict(item))

        elif source in {"vegetation", "population"}:
            selected_items.append(dict(item))

    selected_items.sort(key=lambda item: int(item["channel"]))
    selected_channels = [
        int(item["channel"])
        for item in selected_items
    ]

    expected_count = 4 + 5 * len(selected_levels) + 2 + 1

    if len(selected_channels) != expected_count:
        raise ValueError(
            f"选中特征数量为 {len(selected_channels)}，"
            f"按当前规则应为 {expected_count}。"
        )

    return selected_channels, selected_items, level_mapping


# =============================================================================
# 3. 数据读取与抽样
# =============================================================================
def paths_for_date(date_text: str) -> Tuple[Path, Path]:
    feature_path = (
        FEATURE_DIR / FEATURE_PATTERN.format(date=date_text)
    )
    label_path = (
        LABEL_DIR / LABEL_PATTERN.format(date=date_text)
    )
    return feature_path, label_path


def load_mask() -> np.ndarray:
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"找不到广东掩膜：{MASK_PATH}")

    mask = np.load(MASK_PATH)

    if mask.ndim != 2:
        raise ValueError(
            f"广东掩膜应为二维，当前 shape={mask.shape}"
        )

    return mask.astype(bool)


def load_day_pixels(
    date_text: str,
    selected_channels: Sequence[int],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_path, label_path = paths_for_date(date_text)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"{date_text} 缺少特征：{feature_path}"
        )
    if not label_path.exists():
        raise FileNotFoundError(
            f"{date_text} 缺少标签：{label_path}"
        )

    features = np.load(feature_path, mmap_mode="r")
    confidence = np.load(label_path, mmap_mode="r")

    if features.ndim != 3:
        raise ValueError(
            f"{feature_path} 应为 (C,H,W)，当前 {features.shape}"
        )
    if confidence.ndim == 3 and confidence.shape[0] == 1:
        confidence = confidence[0]
    if confidence.ndim != 2:
        raise ValueError(
            f"{label_path} 应为 (H,W)，当前 {confidence.shape}"
        )
    if features.shape[1:] != mask.shape:
        raise ValueError(
            f"{feature_path} 空间 shape={features.shape[1:]}，"
            f"mask shape={mask.shape}"
        )
    if confidence.shape != mask.shape:
        raise ValueError(
            f"{label_path} shape={confidence.shape}，"
            f"mask shape={mask.shape}"
        )

    # 只取广东区域内像元。
    x = np.asarray(
        features[np.asarray(selected_channels), :, :][:, mask].T,
        dtype=np.float32,
    )

    y = np.asarray(
        confidence[mask] > LABEL_CONFIDENCE_THRESHOLD,
        dtype=np.uint8,
    )

    return x, y


def deterministic_date_seed(date_text: str) -> int:
    return RANDOM_STATE + int(date_text[-6:])


def sample_training_pixels(
    date_text: str,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    positive_indices = np.flatnonzero(y == 1)
    negative_indices = np.flatnonzero(y == 0)

    positive_count = int(positive_indices.size)
    negative_count = int(negative_indices.size)

    rng = np.random.default_rng(
        deterministic_date_seed(date_text)
    )

    if positive_count > 0:
        wanted_negatives = max(
            MIN_NEGATIVES_PER_FIRE_DAY,
            positive_count * NEGATIVE_TO_POSITIVE_RATIO,
        )
    else:
        wanted_negatives = NO_FIRE_DAY_NEGATIVE_SAMPLES

    wanted_negatives = min(
        wanted_negatives,
        MAX_NEGATIVES_PER_DAY,
        negative_count,
    )

    if wanted_negatives > 0:
        selected_negatives = rng.choice(
            negative_indices,
            size=int(wanted_negatives),
            replace=False,
        )
    else:
        selected_negatives = np.empty(0, dtype=np.int64)

    selected_indices = np.concatenate(
        [positive_indices, selected_negatives]
    )

    rng.shuffle(selected_indices)

    sampled_x = np.asarray(
        x[selected_indices],
        dtype=np.float32,
    )
    sampled_y = np.asarray(
        y[selected_indices],
        dtype=np.uint8,
    )

    stats = {
        "full_positive": positive_count,
        "full_negative": negative_count,
        "sampled_positive": int(
            np.count_nonzero(sampled_y == 1)
        ),
        "sampled_negative": int(
            np.count_nonzero(sampled_y == 0)
        ),
    }

    return sampled_x, sampled_y, stats


def prepare_sampled_days(
    dates: Sequence[str],
    selected_channels: Sequence[int],
    mask: np.ndarray,
    log_path: Path,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, int]],
]:
    sampled_x_by_date: Dict[str, np.ndarray] = {}
    sampled_y_by_date: Dict[str, np.ndarray] = {}
    stats_by_date: Dict[str, Dict[str, int]] = {}

    for date_text in dates:
        x_full, y_full = load_day_pixels(
            date_text,
            selected_channels,
            mask,
        )

        sampled_x, sampled_y, stats = sample_training_pixels(
            date_text,
            x_full,
            y_full,
        )

        sampled_x_by_date[date_text] = sampled_x
        sampled_y_by_date[date_text] = sampled_y
        stats_by_date[date_text] = stats

        append_log(
            log_path,
            (
                f"Prepared {date_text}: "
                f"full pos={stats['full_positive']}, "
                f"full neg={stats['full_negative']}, "
                f"sampled pos={stats['sampled_positive']}, "
                f"sampled neg={stats['sampled_negative']}"
            ),
        )

        del x_full
        del y_full
        gc.collect()

    return sampled_x_by_date, sampled_y_by_date, stats_by_date


# =============================================================================
# 4. 预测、阈值和指标
# =============================================================================
def predict_probability_batched(
    model,
    x: np.ndarray,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> np.ndarray:
    probabilities = np.empty(
        x.shape[0],
        dtype=np.float32,
    )

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        batch = x[start:end]

        if hasattr(model, "predict_proba"):
            batch_probability = model.predict_proba(batch)[:, 1]
        elif hasattr(model, "decision_function"):
            score = model.decision_function(batch)
            batch_probability = 1.0 / (1.0 + np.exp(-score))
        else:
            batch_probability = model.predict(batch)

        probabilities[start:end] = np.asarray(
            batch_probability,
            dtype=np.float32,
        )

    return probabilities


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        y_probability,
    )

    if thresholds.size == 0:
        return 0.5, 0.0, precision, recall, thresholds

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    best_index = int(np.nanargmax(f1_values))
    best_threshold = float(thresholds[best_index])
    best_f1 = float(f1_values[best_index])

    return (
        best_threshold,
        best_f1,
        precision,
        recall,
        thresholds,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (
        y_probability >= threshold
    ).astype(np.uint8)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "threshold": float(threshold),
        "sample_count": int(y_true.size),
        "positive_count": int(np.count_nonzero(y_true == 1)),
        "negative_count": int(np.count_nonzero(y_true == 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "cohen_kappa": float(
            cohen_kappa_score(y_true, y_pred)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_probability)
        )
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_probability)
        )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    return metrics


# =============================================================================
# 5. 保存 CSV 和图
# =============================================================================
def save_fold_metrics(
    path: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_threshold_search(
    path: Path,
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
) -> None:
    if thresholds.size == 0:
        return

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    # 阈值数量可能很大，CSV 最多保留约 5000 行。
    if thresholds.size > 5000:
        indices = np.linspace(
            0,
            thresholds.size - 1,
            5000,
            dtype=int,
        )
    else:
        indices = np.arange(thresholds.size)

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["threshold", "precision", "recall", "f1"]
        )

        for index in indices:
            writer.writerow(
                [
                    float(thresholds[index]),
                    float(precision[index]),
                    float(recall[index]),
                    float(f1_values[index]),
                ]
            )


def plot_roc(
    path: Path,
    y_true: np.ndarray,
    y_probability: np.ndarray,
    model_name: str,
) -> None:
    fpr, tpr, _ = roc_curve(
        y_true,
        y_probability,
    )
    roc_auc = roc_auc_score(
        y_true,
        y_probability,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    axis.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(
        f"Date-wise cross-validation ROC\n{model_name}"
    )
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_pr(
    path: Path,
    y_true: np.ndarray,
    y_probability: np.ndarray,
    model_name: str,
) -> None:
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_probability,
    )
    ap = average_precision_score(
        y_true,
        y_probability,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    axis.plot(
        recall,
        precision,
        label=f"AP = {ap:.4f}",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(
        f"Date-wise cross-validation PR curve\n{model_name}"
    )
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_threshold_f1(
    path: Path,
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    best_threshold: float,
    best_f1: float,
    model_name: str,
) -> None:
    if thresholds.size == 0:
        return

    f1_values = (
        2.0 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    figure, axis = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    axis.plot(
        thresholds,
        f1_values,
    )
    axis.axvline(
        best_threshold,
        linestyle="--",
        label=(
            f"Best threshold = {best_threshold:.4f}\n"
            f"Best F1 = {best_f1:.4f}"
        ),
    )
    axis.set_xlabel("Probability threshold")
    axis.set_ylabel("F1 score")
    axis.set_title(
        f"OOF threshold search\n{model_name}"
    )
    axis.legend(loc="best")
    axis.grid(alpha=0.25)

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_confusion(
    path: Path,
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    model_name: str,
) -> None:
    y_pred = (
        y_probability >= threshold
    ).astype(np.uint8)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(
        figsize=(6, 6),
        constrained_layout=True,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No fire", "Fire"],
    )
    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )
    axis.set_title(
        f"OOF confusion matrix\n"
        f"{model_name}, threshold={threshold:.4f}"
    )

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# 6. 单模型训练
# =============================================================================
def train_one_model(
    model_name: str,
    dates: Sequence[str],
    fire_days: Sequence[str],
    selected_channels: Sequence[int],
    sampled_x_by_date: Dict[str, np.ndarray],
    sampled_y_by_date: Dict[str, np.ndarray],
    mask: np.ndarray,
    common_config: Dict[str, object],
    selected_channel_payload: Dict[str, object],
) -> None:
    model_dir = CHECKPOINT_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = model_dir / "training_log.txt"
    if log_path.exists() and OVERWRITE_CHECKPOINT:
        log_path.unlink()

    append_log(
        log_path,
        f"Start training model={model_name}",
    )
    append_log(
        log_path,
        f"Validation fire days={list(fire_days)}",
    )

    oof_y_parts: List[np.ndarray] = []
    oof_probability_parts: List[np.ndarray] = []
    fold_rows: List[Dict[str, object]] = []
    day_slices: Dict[str, Dict[str, int]] = {}
    offset = 0

    for fold_index, validation_day in enumerate(
        fire_days,
        start=1,
    ):
        fold_start = time.time()

        training_dates = [
            date_text
            for date_text in dates
            if date_text != validation_day
        ]

        x_train = np.concatenate(
            [
                sampled_x_by_date[date_text]
                for date_text in training_dates
            ],
            axis=0,
        )
        y_train = np.concatenate(
            [
                sampled_y_by_date[date_text]
                for date_text in training_dates
            ],
            axis=0,
        )

        append_log(
            log_path,
            (
                f"Fold {fold_index}/{len(fire_days)}, "
                f"validation_day={validation_day}, "
                f"train_shape={x_train.shape}, "
                f"train_pos={int(np.count_nonzero(y_train == 1))}, "
                f"train_neg={int(np.count_nonzero(y_train == 0))}"
            ),
        )

        model = build_model(
            model_name,
            random_state=RANDOM_STATE + fold_index,
        )
        model.fit(x_train, y_train)

        x_validation, y_validation = load_day_pixels(
            validation_day,
            selected_channels,
            mask,
        )

        validation_probability = predict_probability_batched(
            model,
            x_validation,
        )

        (
            fold_threshold,
            fold_best_f1,
            _,
            _,
            _,
        ) = find_best_f1_threshold(
            y_validation,
            validation_probability,
        )

        fold_metrics = compute_metrics(
            y_validation,
            validation_probability,
            fold_threshold,
        )

        fold_row: Dict[str, object] = {
            "fold": fold_index,
            "validation_day": validation_day,
            "training_date_count": len(training_dates),
            "training_sample_count": int(x_train.shape[0]),
            "validation_sample_count": int(x_validation.shape[0]),
            "validation_positive_count": int(
                np.count_nonzero(y_validation == 1)
            ),
            "validation_negative_count": int(
                np.count_nonzero(y_validation == 0)
            ),
            "best_threshold": float(fold_threshold),
            "best_f1": float(fold_best_f1),
            "precision": fold_metrics["precision"],
            "recall": fold_metrics["recall"],
            "specificity": fold_metrics["specificity"],
            "roc_auc": fold_metrics["roc_auc"],
            "average_precision": fold_metrics["average_precision"],
            "elapsed_seconds": float(time.time() - fold_start),
        }
        fold_rows.append(fold_row)

        start_offset = offset
        end_offset = offset + y_validation.size
        day_slices[validation_day] = {
            "start": int(start_offset),
            "end": int(end_offset),
        }
        offset = end_offset

        oof_y_parts.append(
            np.asarray(y_validation, dtype=np.uint8)
        )
        oof_probability_parts.append(
            np.asarray(
                validation_probability,
                dtype=np.float32,
            )
        )

        append_log(
            log_path,
            (
                f"Fold {fold_index} done: "
                f"threshold={fold_threshold:.6f}, "
                f"F1={fold_metrics['f1']:.6f}, "
                f"precision={fold_metrics['precision']:.6f}, "
                f"recall={fold_metrics['recall']:.6f}, "
                f"ROC_AUC={fold_metrics['roc_auc']:.6f}"
            ),
        )

        del model
        del x_train
        del y_train
        del x_validation
        del y_validation
        del validation_probability
        gc.collect()

    oof_y = np.concatenate(
        oof_y_parts,
        axis=0,
    )
    oof_probability = np.concatenate(
        oof_probability_parts,
        axis=0,
    )

    (
        best_threshold,
        best_f1,
        precision,
        recall,
        thresholds,
    ) = find_best_f1_threshold(
        oof_y,
        oof_probability,
    )

    metrics = compute_metrics(
        oof_y,
        oof_probability,
        best_threshold,
    )
    metrics["model_name"] = model_name
    metrics["validation_days"] = list(fire_days)
    metrics["validation_day_count"] = len(fire_days)

    append_log(
        log_path,
        (
            f"Global OOF threshold={best_threshold:.8f}, "
            f"F1={metrics['f1']:.8f}, "
            f"precision={metrics['precision']:.8f}, "
            f"recall={metrics['recall']:.8f}, "
            f"ROC_AUC={metrics['roc_auc']:.8f}, "
            f"AP={metrics['average_precision']:.8f}"
        ),
    )

    save_fold_metrics(
        model_dir / "fold_metrics.csv",
        fold_rows,
    )
    save_threshold_search(
        model_dir / "threshold_search.csv",
        precision,
        recall,
        thresholds,
    )
    save_json(
        model_dir / "metrics.json",
        metrics,
    )
    save_json(
        model_dir / "threshold.json",
        {
            "model_name": model_name,
            "threshold": float(best_threshold),
            "objective": "maximum OOF F1",
            "best_oof_f1": float(best_f1),
        },
    )
    save_json(
        model_dir / "selected_channels.json",
        selected_channel_payload,
    )
    save_json(
        model_dir / "training_config.json",
        common_config,
    )
    save_json(
        model_dir / "oof_day_slices.json",
        day_slices,
    )

    if SAVE_OOF_PREDICTIONS:
        np.savez_compressed(
            model_dir / "oof_predictions.npz",
            y_true=oof_y.astype(np.uint8),
            y_probability=oof_probability.astype(np.float32),
        )

    plot_roc(
        model_dir / "roc_curve.png",
        oof_y,
        oof_probability,
        model_name,
    )
    plot_pr(
        model_dir / "pr_curve.png",
        oof_y,
        oof_probability,
        model_name,
    )
    plot_threshold_f1(
        model_dir / "threshold_f1_curve.png",
        precision,
        recall,
        thresholds,
        best_threshold,
        best_f1,
        model_name,
    )
    plot_confusion(
        model_dir / "confusion_matrix.png",
        oof_y,
        oof_probability,
        best_threshold,
        model_name,
    )

    # 使用全部日期抽样数据训练最终模型。
    x_all = np.concatenate(
        [sampled_x_by_date[date_text] for date_text in dates],
        axis=0,
    )
    y_all = np.concatenate(
        [sampled_y_by_date[date_text] for date_text in dates],
        axis=0,
    )

    append_log(
        log_path,
        (
            f"Fit final model on all dates: "
            f"shape={x_all.shape}, "
            f"pos={int(np.count_nonzero(y_all == 1))}, "
            f"neg={int(np.count_nonzero(y_all == 0))}"
        ),
    )

    final_model = build_model(
        model_name,
        random_state=RANDOM_STATE,
    )
    final_model.fit(x_all, y_all)

    joblib.dump(
        final_model,
        model_dir / "model.joblib",
        compress=3,
    )

    append_log(
        log_path,
        f"Saved final model: {model_dir / 'model.joblib'}",
    )

    del final_model
    del x_all
    del y_all
    del oof_y
    del oof_probability
    gc.collect()


# =============================================================================
# 7. 主流程
# =============================================================================
def main() -> None:
    validate_model_names(ENABLED_MODELS)

    dates = list(
        date_range(START_DATE, END_DATE)
    )

    if not dates:
        raise ValueError("没有训练日期。")

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CHANNEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到通道元数据：{CHANNEL_METADATA_PATH}"
        )

    metadata = load_json(
        CHANNEL_METADATA_PATH
    )

    (
        selected_channels,
        selected_items,
        level_mapping,
    ) = select_feature_channels(metadata)

    mask = load_mask()

    selected_channel_payload = {
        "selected_channel_indices": selected_channels,
        "selected_channel_count": len(selected_channels),
        "selected_channel_metadata": selected_items,
        "requested_pressure_levels_hpa": REQUESTED_PRESSURE_LEVELS,
        "resolved_pressure_level_mapping_hpa": {
            str(key): value
            for key, value in level_mapping.items()
        },
    }

    common_config = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "dates": dates,
        "feature_dir": str(FEATURE_DIR),
        "feature_pattern": FEATURE_PATTERN,
        "label_dir": str(LABEL_DIR),
        "label_pattern": LABEL_PATTERN,
        "mask_path": str(MASK_PATH),
        "channel_metadata_path": str(CHANNEL_METADATA_PATH),
        "label_confidence_threshold": LABEL_CONFIDENCE_THRESHOLD,
        "requested_pressure_levels_hpa": REQUESTED_PRESSURE_LEVELS,
        "resolved_pressure_level_mapping_hpa": level_mapping,
        "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
        "min_negatives_per_fire_day": MIN_NEGATIVES_PER_FIRE_DAY,
        "max_negatives_per_day": MAX_NEGATIVES_PER_DAY,
        "no_fire_day_negative_samples": NO_FIRE_DAY_NEGATIVE_SAMPLES,
        "random_state": RANDOM_STATE,
        "enabled_models": ENABLED_MODELS,
    }

    # 公共数据准备日志。
    preparation_log = (
        CHECKPOINT_ROOT / "data_preparation_log.txt"
    )
    if preparation_log.exists() and OVERWRITE_CHECKPOINT:
        preparation_log.unlink()

    append_log(
        preparation_log,
        f"Selected channels={selected_channels}",
    )
    append_log(
        preparation_log,
        f"Pressure level mapping={level_mapping}",
    )

    (
        sampled_x_by_date,
        sampled_y_by_date,
        stats_by_date,
    ) = prepare_sampled_days(
        dates,
        selected_channels,
        mask,
        preparation_log,
    )

    save_json(
        CHECKPOINT_ROOT / "daily_sampling_stats.json",
        stats_by_date,
    )
    save_json(
        CHECKPOINT_ROOT / "selected_channels.json",
        selected_channel_payload,
    )
    save_json(
        CHECKPOINT_ROOT / "training_config.json",
        common_config,
    )

    fire_days = [
        date_text
        for date_text in dates
        if stats_by_date[date_text]["full_positive"] > 0
    ]

    if len(fire_days) < 2:
        raise ValueError(
            "有火点日期少于 2 天，无法进行按日期交叉验证。"
        )

    append_log(
        preparation_log,
        f"Fire days used as validation folds={fire_days}",
    )

    for model_name in ENABLED_MODELS:
        train_one_model(
            model_name=model_name,
            dates=dates,
            fire_days=fire_days,
            selected_channels=selected_channels,
            sampled_x_by_date=sampled_x_by_date,
            sampled_y_by_date=sampled_y_by_date,
            mask=mask,
            common_config=common_config,
            selected_channel_payload=selected_channel_payload,
        )

    print("\n[DONE] All models trained.")
    print("Checkpoint root:", CHECKPOINT_ROOT.resolve())


if __name__ == "__main__":
    main()