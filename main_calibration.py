#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用已有 checkpoint 的日期 OOF 预测进行后校准。

为什么不重新训练基础模型
----------------------
原训练脚本的基础分类器使用了负样本抽样，而保存的 OOF 概率是在完整验证日的
全部广东像元上计算的。概率校准需要尽量接近真实部署时的类别比例，因此直接使用
完整验证日 OOF 概率拟合后校准器，比在抽样训练集上重新做概率校准更合适。

方法
----
1. 读取 checkpoint/<model_name>/oof_predictions.npz。
2. 根据 oof_day_slices.json 进行留一日期交叉校准。
3. 比较：
       sigmoid
       isotonic
4. 主要按交叉校准 Brier score 选择方法；
   log loss 和 ECE 作为次级判据。
5. 在全部 OOF 概率上拟合最终校准器。
6. 使用该校准器封装已有的最终 model.joblib。
7. 使用交叉校准后的 OOF 概率重新搜索最大 F1 阈值。
8. 输出与原 checkpoint 基本一致的目录结构。

输出
----
checkpoint_calibration/
├── calibration_summary.csv
├── calibration_summary.png
├── <model_name>/
│   ├── model.joblib
│   ├── threshold.json
│   ├── metrics.json
│   ├── fold_metrics.csv
│   ├── threshold_search.csv
│   ├── selected_channels.json
│   ├── training_config.json
│   ├── training_log.txt
│   ├── oof_predictions.npz
│   ├── oof_day_slices.json
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── threshold_f1_curve.png
│   ├── confusion_matrix.png
│   ├── calibration.json
│   ├── calibration_method_comparison.csv
│   ├── calibration_curve.csv
│   ├── calibration_curve.png
│   ├── calibration_metric_comparison.png
│   └── calibrator.joblib
└── ...

使用
----
把现有预测脚本中的：
    CHECKPOINT_ROOT = Path("./checkpoint")
改为：
    CHECKPOINT_ROOT = Path("./checkpoint_calibration")

其余预测代码不需要修改。

前置文件
--------
请把 companion 文件放到：
    ./model/calibrated_probability_model.py
"""

from __future__ import annotations

import csv
import gc
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from model.calibrated_probability_model import (
        CalibratedProbabilityModel,
        build_probability_calibrator,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "缺少 ./model/calibrated_probability_model.py。"
        "请把配套文件放入项目 model 目录后再运行。"
    ) from exc


# =============================================================================
# 0. 配置
# =============================================================================
SOURCE_CHECKPOINT_ROOT = Path("./checkpoint")
OUTPUT_CHECKPOINT_ROOT = Path("./checkpoint_calibration")

ENABLED_MODELS = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
]

CANDIDATE_METHODS = [
    "sigmoid",
    "isotonic",
]

# 阈值仍以校准后的交叉验证 OOF 概率最大化 F1。
THRESHOLD_OBJECTIVE = "maximum cross-fitted calibrated OOF F1"

# 校准拟合样本很多时，使用带权抽样降低时间与内存。
# 全部正样本优先保留，负样本被抽样后使用 sample_weight 恢复原始比例。
MAX_CALIBRATION_FIT_SAMPLES = 2_000_000

# 概率校准曲线使用固定的 0～1 等宽分箱。
CALIBRATION_BIN_COUNT = 15

RANDOM_STATE = 42
OVERWRITE_OUTPUT = True
COPY_SOURCE_CHECKPOINT = True
FIGURE_DPI = 250
PROBABILITY_EPS = 1e-7


# =============================================================================
# 1. 通用工具
# =============================================================================
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
            allow_nan=True,
        )


def append_log(path: Path, message: str) -> None:
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    line = f"[{timestamp}] {message}"
    print(line)

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(line + "\n")


def copy_checkpoint_structure() -> None:
    if not SOURCE_CHECKPOINT_ROOT.exists():
        raise FileNotFoundError(
            f"找不到源 checkpoint："
            f"{SOURCE_CHECKPOINT_ROOT}"
        )

    if OUTPUT_CHECKPOINT_ROOT.exists():
        if OVERWRITE_OUTPUT:
            shutil.rmtree(
                OUTPUT_CHECKPOINT_ROOT
            )
        else:
            raise FileExistsError(
                f"输出目录已存在："
                f"{OUTPUT_CHECKPOINT_ROOT}"
            )

    if COPY_SOURCE_CHECKPOINT:
        shutil.copytree(
            SOURCE_CHECKPOINT_ROOT,
            OUTPUT_CHECKPOINT_ROOT,
        )
    else:
        OUTPUT_CHECKPOINT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_probability(
    probability: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(
        probability,
        dtype=np.float64,
    ).reshape(-1)

    probability = np.nan_to_num(
        probability,
        nan=0.5,
        posinf=1.0 - PROBABILITY_EPS,
        neginf=PROBABILITY_EPS,
    )

    return np.clip(
        probability,
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )


# =============================================================================
# 2. 指标
# =============================================================================
def expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = CALIBRATION_BIN_COUNT,
) -> float:
    table = calibration_table(
        y_true,
        probability,
        n_bins=n_bins,
    )

    if not table:
        return float("nan")

    total = sum(
        row["count"]
        for row in table
    )

    if total == 0:
        return float("nan")

    return float(
        sum(
            row["count"]
            * abs(
                row["observed_rate"]
                - row["mean_probability"]
            )
            for row in table
        )
        / total
    )


def maximum_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = CALIBRATION_BIN_COUNT,
) -> float:
    table = calibration_table(
        y_true,
        probability,
        n_bins=n_bins,
    )

    if not table:
        return float("nan")

    return float(
        max(
            abs(
                row["observed_rate"]
                - row["mean_probability"]
            )
            for row in table
        )
    )


def calibration_table(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = CALIBRATION_BIN_COUNT,
) -> List[Dict[str, float]]:
    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    ).reshape(-1)

    probability = validate_probability(
        probability
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_index = np.digitize(
        probability,
        edges[1:-1],
        right=False,
    )

    rows: List[Dict[str, float]] = []

    for index in range(n_bins):
        mask = bin_index == index
        count = int(mask.sum())

        if count == 0:
            continue

        rows.append(
            {
                "bin": int(index + 1),
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "count": count,
                "mean_probability": float(
                    probability[mask].mean()
                ),
                "observed_rate": float(
                    y_true[mask].mean()
                ),
            }
        )

    return rows


def probability_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    ).reshape(-1)

    probability = validate_probability(
        probability
    )

    metrics = {
        "brier_score": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack(
                    [
                        1.0 - probability,
                        probability,
                    ]
                ),
                labels=[0, 1],
            )
        ),
        "ece": float(
            expected_calibration_error(
                y_true,
                probability,
            )
        ),
        "mce": float(
            maximum_calibration_error(
                y_true,
                probability,
            )
        ),
    }

    if np.unique(y_true).size == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )
        metrics["average_precision"] = float(
            average_precision_score(
                y_true,
                probability,
            )
        )
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float(
            "nan"
        )

    return metrics


def find_best_f1_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> Tuple[
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )

    if thresholds.size == 0:
        return (
            0.5,
            0.0,
            precision,
            recall,
            thresholds,
        )

    f1_values = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(
            precision[:-1]
            + recall[:-1],
            1e-12,
        )
    )

    best_index = int(
        np.nanargmax(f1_values)
    )

    return (
        float(thresholds[best_index]),
        float(f1_values[best_index]),
        precision,
        recall,
        thresholds,
    )


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    )
    probability = validate_probability(
        probability
    )
    y_pred = (
        probability >= threshold
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

    metrics: Dict[str, object] = {
        "threshold": float(threshold),
        "sample_count": int(y_true.size),
        "positive_count": int(
            np.count_nonzero(y_true == 1)
        ),
        "negative_count": int(
            np.count_nonzero(y_true == 0)
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
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
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    metrics.update(
        probability_metrics(
            y_true,
            probability,
        )
    )

    return metrics


# =============================================================================
# 3. 校准拟合
# =============================================================================
def weighted_calibration_sample(
    probability: np.ndarray,
    y_true: np.ndarray,
    max_samples: int,
    random_state: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
]:
    probability = validate_probability(
        probability
    )
    y_true = np.asarray(
        y_true,
        dtype=np.uint8,
    )

    sample_count = y_true.size

    if (
        max_samples <= 0
        or sample_count <= max_samples
    ):
        return probability, y_true, None

    rng = np.random.default_rng(
        random_state
    )

    positive_indices = np.flatnonzero(
        y_true == 1
    )
    negative_indices = np.flatnonzero(
        y_true == 0
    )

    # 优先保留全部正样本。
    if positive_indices.size >= max_samples:
        selected_positive_count = max(
            2,
            max_samples // 2,
        )
    else:
        selected_positive_count = (
            positive_indices.size
        )

    selected_negative_count = (
        max_samples
        - selected_positive_count
    )

    selected_positive_count = min(
        selected_positive_count,
        positive_indices.size,
    )
    selected_negative_count = min(
        selected_negative_count,
        negative_indices.size,
    )

    selected_positive = rng.choice(
        positive_indices,
        size=selected_positive_count,
        replace=False,
    )
    selected_negative = rng.choice(
        negative_indices,
        size=selected_negative_count,
        replace=False,
    )

    selected = np.concatenate(
        [
            selected_positive,
            selected_negative,
        ]
    )
    rng.shuffle(selected)

    sample_weight = np.ones(
        selected.size,
        dtype=np.float64,
    )

    positive_mask = (
        y_true[selected] == 1
    )
    negative_mask = ~positive_mask

    if selected_positive_count > 0:
        sample_weight[positive_mask] = (
            positive_indices.size
            / selected_positive_count
        )

    if selected_negative_count > 0:
        sample_weight[negative_mask] = (
            negative_indices.size
            / selected_negative_count
        )

    return (
        probability[selected],
        y_true[selected],
        sample_weight,
    )


def fit_calibrator(
    method: str,
    probability: np.ndarray,
    y_true: np.ndarray,
    random_state: int,
):
    (
        sampled_probability,
        sampled_y,
        sample_weight,
    ) = weighted_calibration_sample(
        probability=probability,
        y_true=y_true,
        max_samples=MAX_CALIBRATION_FIT_SAMPLES,
        random_state=random_state,
    )

    calibrator = build_probability_calibrator(
        method,
        random_state=random_state,
    )

    calibrator.fit(
        sampled_probability,
        sampled_y,
        sample_weight=sample_weight,
    )

    return calibrator


def ordered_day_slices(
    day_slices: Dict[str, Dict[str, int]],
    sample_count: int,
) -> List[
    Tuple[str, int, int]
]:
    rows = []

    for date_text, bounds in (
        day_slices.items()
    ):
        start = int(bounds["start"])
        end = int(bounds["end"])

        if not (
            0 <= start < end <= sample_count
        ):
            raise ValueError(
                f"{date_text} 的 OOF 切片非法："
                f"start={start}, end={end}, "
                f"sample_count={sample_count}"
            )

        rows.append(
            (date_text, start, end)
        )

    rows.sort(
        key=lambda item: item[1]
    )

    coverage = np.zeros(
        sample_count,
        dtype=np.uint8,
    )

    for _, start, end in rows:
        coverage[start:end] += 1

    if not np.all(coverage == 1):
        raise ValueError(
            "oof_day_slices.json 未对 OOF 样本"
            "形成恰好一次的完整覆盖。"
        )

    if len(rows) < 2:
        raise ValueError(
            "至少需要两个 OOF 验证日期才能进行"
            "留一日期交叉校准。"
        )

    return rows


def cross_fitted_calibration(
    method: str,
    y_true: np.ndarray,
    raw_probability: np.ndarray,
    day_slices: Sequence[
        Tuple[str, int, int]
    ],
    random_state: int,
) -> Tuple[
    np.ndarray,
    List[Dict[str, object]],
]:
    sample_count = y_true.size

    calibrated = np.full(
        sample_count,
        np.nan,
        dtype=np.float64,
    )

    fold_rows: List[
        Dict[str, object]
    ] = []

    all_indices = np.arange(
        sample_count
    )

    for fold_index, (
        date_text,
        start,
        end,
    ) in enumerate(
        day_slices,
        start=1,
    ):
        validation_indices = np.arange(
            start,
            end,
        )

        training_mask = np.ones(
            sample_count,
            dtype=bool,
        )
        training_mask[
            validation_indices
        ] = False

        training_indices = all_indices[
            training_mask
        ]

        if (
            np.unique(
                y_true[training_indices]
            ).size
            < 2
        ):
            raise ValueError(
                f"{date_text} 留出后，校准训练数据"
                "没有同时包含两个类别。"
            )

        calibrator = fit_calibrator(
            method=method,
            probability=raw_probability[
                training_indices
            ],
            y_true=y_true[
                training_indices
            ],
            random_state=(
                random_state
                + fold_index
            ),
        )

        fold_probability = calibrator.predict(
            raw_probability[
                validation_indices
            ]
        )

        calibrated[
            validation_indices
        ] = fold_probability

        (
            fold_threshold,
            fold_best_f1,
            _,
            _,
            _,
        ) = find_best_f1_threshold(
            y_true[validation_indices],
            fold_probability,
        )

        fold_metric = (
            classification_metrics(
                y_true[
                    validation_indices
                ],
                fold_probability,
                fold_threshold,
            )
        )

        fold_rows.append(
            {
                "fold": fold_index,
                "validation_day": date_text,
                "calibration_method": method,
                "calibration_training_samples": int(
                    training_indices.size
                ),
                "validation_samples": int(
                    validation_indices.size
                ),
                "validation_positive_count": int(
                    np.count_nonzero(
                        y_true[
                            validation_indices
                        ]
                        == 1
                    )
                ),
                "validation_negative_count": int(
                    np.count_nonzero(
                        y_true[
                            validation_indices
                        ]
                        == 0
                    )
                ),
                "best_threshold": float(
                    fold_threshold
                ),
                "best_f1": float(
                    fold_best_f1
                ),
                "precision": fold_metric[
                    "precision"
                ],
                "recall": fold_metric[
                    "recall"
                ],
                "specificity": fold_metric[
                    "specificity"
                ],
                "brier_score": fold_metric[
                    "brier_score"
                ],
                "log_loss": fold_metric[
                    "log_loss"
                ],
                "ece": fold_metric["ece"],
                "roc_auc": fold_metric[
                    "roc_auc"
                ],
                "average_precision": fold_metric[
                    "average_precision"
                ],
            }
        )

        del calibrator
        gc.collect()

    if not np.all(
        np.isfinite(calibrated)
    ):
        raise RuntimeError(
            "交叉校准后仍存在未赋值或非有限概率。"
        )

    return calibrated, fold_rows


# =============================================================================
# 4. 文件保存
# =============================================================================
def save_csv(
    path: Path,
    rows: Sequence[
        Dict[str, object]
    ],
) -> None:
    if not rows:
        return

    fieldnames: List[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

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
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(
            precision[:-1]
            + recall[:-1],
            1e-12,
        )
    )

    if thresholds.size > 5000:
        indices = np.linspace(
            0,
            thresholds.size - 1,
            5000,
            dtype=int,
        )
    else:
        indices = np.arange(
            thresholds.size
        )

    rows = []

    for index in indices:
        rows.append(
            {
                "threshold": float(
                    thresholds[index]
                ),
                "precision": float(
                    precision[index]
                ),
                "recall": float(
                    recall[index]
                ),
                "f1": float(
                    f1_values[index]
                ),
            }
        )

    save_csv(path, rows)


# =============================================================================
# 5. 绘图
# =============================================================================
def plot_roc(
    path: Path,
    y_true: np.ndarray,
    probability: np.ndarray,
    model_name: str,
) -> None:
    fpr, tpr, _ = roc_curve(
        y_true,
        probability,
    )
    auc = roc_auc_score(
        y_true,
        probability,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    axis.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}",
    )
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )
    axis.set_xlabel(
        "False positive rate"
    )
    axis.set_ylabel(
        "True positive rate"
    )
    axis.set_title(
        f"Calibrated OOF ROC\n{model_name}"
    )
    axis.grid(alpha=0.25)
    axis.legend(
        loc="lower right"
    )

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_pr(
    path: Path,
    y_true: np.ndarray,
    probability: np.ndarray,
    model_name: str,
) -> None:
    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            probability,
        )
    )
    ap = average_precision_score(
        y_true,
        probability,
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
        f"Calibrated OOF PR curve\n"
        f"{model_name}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

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
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(
            precision[:-1]
            + recall[:-1],
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
            f"Best threshold = "
            f"{best_threshold:.6f}\n"
            f"Best F1 = {best_f1:.4f}"
        ),
    )
    axis.set_xlabel(
        "Calibrated probability threshold"
    )
    axis.set_ylabel("F1 score")
    axis.set_title(
        f"Calibrated OOF threshold search\n"
        f"{model_name}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_confusion(
    path: Path,
    y_true: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    model_name: str,
) -> None:
    y_pred = (
        probability >= threshold
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
        display_labels=[
            "No fire",
            "Fire",
        ],
    )
    display.plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )
    axis.set_title(
        f"Calibrated OOF confusion matrix\n"
        f"{model_name}, "
        f"threshold={threshold:.6f}"
    )

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_calibration_curve(
    path: Path,
    raw_table: Sequence[
        Dict[str, float]
    ],
    calibrated_table: Sequence[
        Dict[str, float]
    ],
    model_name: str,
    raw_metrics: Dict[str, float],
    calibrated_metrics: Dict[
        str,
        float,
    ],
) -> None:
    figure, axis = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.plot(
        [
            row["mean_probability"]
            for row in raw_table
        ],
        [
            row["observed_rate"]
            for row in raw_table
        ],
        marker="o",
        label=(
            "Raw "
            f"(Brier={raw_metrics['brier_score']:.6f}, "
            f"ECE={raw_metrics['ece']:.6f})"
        ),
    )

    axis.plot(
        [
            row["mean_probability"]
            for row in calibrated_table
        ],
        [
            row["observed_rate"]
            for row in calibrated_table
        ],
        marker="o",
        label=(
            "Calibrated "
            f"(Brier={calibrated_metrics['brier_score']:.6f}, "
            f"ECE={calibrated_metrics['ece']:.6f})"
        ),
    )

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel(
        "Mean predicted probability"
    )
    axis.set_ylabel(
        "Observed fire frequency"
    )
    axis.set_title(
        f"OOF reliability diagram\n"
        f"{model_name}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_method_comparison(
    path: Path,
    rows: Sequence[
        Dict[str, object]
    ],
    model_name: str,
) -> None:
    methods = [
        str(row["method"])
        for row in rows
    ]
    metric_names = [
        "brier_score",
        "log_loss",
        "ece",
    ]

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(14, 5),
        constrained_layout=True,
    )

    for axis, metric in zip(
        axes,
        metric_names,
    ):
        values = [
            float(row[metric])
            for row in rows
        ]
        axis.bar(
            methods,
            values,
        )
        axis.set_title(metric)
        axis.tick_params(
            axis="x",
            rotation=20,
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    figure.suptitle(
        f"Calibration method comparison - "
        f"{model_name}"
    )

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# 6. 单模型校准
# =============================================================================
def calibrate_one_model(
    model_name: str,
) -> Dict[str, object]:
    source_dir = (
        SOURCE_CHECKPOINT_ROOT
        / model_name
    )
    output_dir = (
        OUTPUT_CHECKPOINT_ROOT
        / model_name
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    required = [
        source_dir / "model.joblib",
        source_dir / "oof_predictions.npz",
        source_dir / "oof_day_slices.json",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"{model_name} 缺少文件：{path}"
            )

    log_path = (
        output_dir
        / "training_log.txt"
    )

    # 源日志已被复制，追加校准部分。
    append_log(
        log_path,
        "=" * 70,
    )
    append_log(
        log_path,
        f"Start probability calibration: "
        f"model={model_name}",
    )

    with np.load(
        source_dir
        / "oof_predictions.npz"
    ) as payload:
        y_true = np.asarray(
            payload["y_true"],
            dtype=np.uint8,
        )
        raw_probability = (
            validate_probability(
                payload["y_probability"]
            )
        )

    day_slices_payload = load_json(
        source_dir
        / "oof_day_slices.json"
    )

    day_slices = ordered_day_slices(
        day_slices_payload,
        sample_count=y_true.size,
    )

    raw_metrics = probability_metrics(
        y_true,
        raw_probability,
    )

    append_log(
        log_path,
        (
            "Raw OOF: "
            f"Brier={raw_metrics['brier_score']:.8f}, "
            f"LogLoss={raw_metrics['log_loss']:.8f}, "
            f"ECE={raw_metrics['ece']:.8f}, "
            f"AUC={raw_metrics['roc_auc']:.8f}, "
            f"AP={raw_metrics['average_precision']:.8f}"
        ),
    )

    method_probabilities: Dict[
        str,
        np.ndarray,
    ] = {}
    method_fold_rows: Dict[
        str,
        List[Dict[str, object]],
    ] = {}
    comparison_rows: List[
        Dict[str, object]
    ] = [
        {
            "method": "raw",
            **raw_metrics,
            "selected": False,
        }
    ]

    for method_index, method in enumerate(
        CANDIDATE_METHODS,
        start=1,
    ):
        start_time = time.time()

        append_log(
            log_path,
            f"Cross-fit calibration method="
            f"{method}",
        )

        calibrated_probability, fold_rows = (
            cross_fitted_calibration(
                method=method,
                y_true=y_true,
                raw_probability=(
                    raw_probability
                ),
                day_slices=day_slices,
                random_state=(
                    RANDOM_STATE
                    + method_index * 1000
                ),
            )
        )

        metrics = probability_metrics(
            y_true,
            calibrated_probability,
        )

        method_probabilities[method] = (
            calibrated_probability
        )
        method_fold_rows[method] = (
            fold_rows
        )

        comparison_rows.append(
            {
                "method": method,
                **metrics,
                "elapsed_seconds": float(
                    time.time()
                    - start_time
                ),
                "selected": False,
            }
        )

        append_log(
            log_path,
            (
                f"{method}: "
                f"Brier={metrics['brier_score']:.8f}, "
                f"LogLoss={metrics['log_loss']:.8f}, "
                f"ECE={metrics['ece']:.8f}, "
                f"AUC={metrics['roc_auc']:.8f}, "
                f"AP={metrics['average_precision']:.8f}"
            ),
        )

    calibrated_candidates = [
        row
        for row in comparison_rows
        if row["method"]
        in CANDIDATE_METHODS
    ]

    selected_row = min(
        calibrated_candidates,
        key=lambda row: (
            float(row["brier_score"]),
            float(row["log_loss"]),
            float(row["ece"]),
        ),
    )

    selected_method = str(
        selected_row["method"]
    )

    for row in comparison_rows:
        row["selected"] = (
            row["method"]
            == selected_method
        )

    cross_fitted_probability = (
        method_probabilities[
            selected_method
        ]
    )

    calibrated_probability_metrics = (
        probability_metrics(
            y_true,
            cross_fitted_probability,
        )
    )

    append_log(
        log_path,
        (
            f"Selected calibration method="
            f"{selected_method}; "
            "selection objective=min "
            "(Brier, LogLoss, ECE)"
        ),
    )

    # 在全部完整 OOF 样本上拟合部署用校准器。
    final_calibrator = fit_calibrator(
        method=selected_method,
        probability=raw_probability,
        y_true=y_true,
        random_state=RANDOM_STATE,
    )

    base_model = joblib.load(
        source_dir / "model.joblib"
    )

    wrapped_model = (
        CalibratedProbabilityModel(
            base_model=base_model,
            calibrator=final_calibrator,
            calibration_method=(
                selected_method
            ),
            metadata={
                "source_checkpoint": str(
                    source_dir
                ),
                "calibration_source": (
                    "date-wise OOF full-day "
                    "probabilities"
                ),
                "method_selection": (
                    "leave-one-date-out "
                    "cross-fitted calibration"
                ),
            },
        )
    )

    joblib.dump(
        wrapped_model,
        output_dir / "model.joblib",
        compress=3,
    )
    joblib.dump(
        final_calibrator,
        output_dir / "calibrator.joblib",
        compress=3,
    )

    (
        best_threshold,
        best_f1,
        precision,
        recall,
        thresholds,
    ) = find_best_f1_threshold(
        y_true,
        cross_fitted_probability,
    )

    metrics = classification_metrics(
        y_true,
        cross_fitted_probability,
        best_threshold,
    )

    metrics.update(
        {
            "model_name": model_name,
            "calibrated": True,
            "calibration_method": (
                selected_method
            ),
            "calibration_evaluation": (
                "leave-one-validation-day-out "
                "cross-fitted calibration"
            ),
            "validation_days": [
                date_text
                for (
                    date_text,
                    _,
                    _,
                ) in day_slices
            ],
            "validation_day_count": len(
                day_slices
            ),
            "raw_brier_score": (
                raw_metrics[
                    "brier_score"
                ]
            ),
            "raw_log_loss": (
                raw_metrics[
                    "log_loss"
                ]
            ),
            "raw_ece": raw_metrics[
                "ece"
            ],
            "brier_improvement": float(
                raw_metrics[
                    "brier_score"
                ]
                - calibrated_probability_metrics[
                    "brier_score"
                ]
            ),
            "log_loss_improvement": float(
                raw_metrics[
                    "log_loss"
                ]
                - calibrated_probability_metrics[
                    "log_loss"
                ]
            ),
            "ece_improvement": float(
                raw_metrics["ece"]
                - calibrated_probability_metrics[
                    "ece"
                ]
            ),
            "metrics_at_probability_0_5": (
                classification_metrics(
                    y_true,
                    cross_fitted_probability,
                    0.5,
                )
            ),
        }
    )

    save_json(
        output_dir / "metrics.json",
        metrics,
    )

    save_json(
        output_dir / "threshold.json",
        {
            "model_name": model_name,
            "threshold": float(
                best_threshold
            ),
            "objective": (
                THRESHOLD_OBJECTIVE
            ),
            "best_oof_f1": float(
                best_f1
            ),
            "calibrated": True,
            "calibration_method": (
                selected_method
            ),
            "probability_interpretation": (
                "Post-calibrated estimated "
                "fire probability"
            ),
            "reference_probability_threshold": (
                0.5
            ),
        },
    )

    calibration_metadata = {
        "model_name": model_name,
        "source_checkpoint_root": str(
            SOURCE_CHECKPOINT_ROOT
        ),
        "output_checkpoint_root": str(
            OUTPUT_CHECKPOINT_ROOT
        ),
        "candidate_methods": list(
            CANDIDATE_METHODS
        ),
        "selected_method": (
            selected_method
        ),
        "method_selection_objective": (
            "minimum leave-one-date-out "
            "cross-fitted Brier score; "
            "log loss and ECE as tie breakers"
        ),
        "calibration_training_data": (
            "full-day date-wise OOF "
            "probabilities and binary labels"
        ),
        "base_model_retrained": False,
        "base_model_file": (
            "copied existing final model and "
            "wrapped with fitted calibrator"
        ),
        "deployment_model_file": (
            "model.joblib"
        ),
        "calibrator_file": (
            "calibrator.joblib"
        ),
        "threshold_source": (
            THRESHOLD_OBJECTIVE
        ),
        "raw_probability_metrics": (
            raw_metrics
        ),
        "cross_fitted_calibrated_metrics": (
            calibrated_probability_metrics
        ),
    }

    save_json(
        output_dir / "calibration.json",
        calibration_metadata,
    )

    save_csv(
        output_dir
        / "calibration_method_comparison.csv",
        comparison_rows,
    )

    fold_rows = method_fold_rows[
        selected_method
    ]

    # 在 fold_metrics.csv 中同时记录全局阈值表现。
    for row, (
        date_text,
        start,
        end,
    ) in zip(
        fold_rows,
        day_slices,
    ):
        global_metrics = (
            classification_metrics(
                y_true[start:end],
                cross_fitted_probability[
                    start:end
                ],
                best_threshold,
            )
        )
        row[
            "global_threshold"
        ] = float(best_threshold)
        row[
            "f1_at_global_threshold"
        ] = global_metrics["f1"]
        row[
            "precision_at_global_threshold"
        ] = global_metrics[
            "precision"
        ]
        row[
            "recall_at_global_threshold"
        ] = global_metrics["recall"]
        row[
            "specificity_at_global_threshold"
        ] = global_metrics[
            "specificity"
        ]

    save_csv(
        output_dir / "fold_metrics.csv",
        fold_rows,
    )

    save_threshold_search(
        output_dir
        / "threshold_search.csv",
        precision,
        recall,
        thresholds,
    )

    np.savez_compressed(
        output_dir
        / "oof_predictions.npz",
        y_true=y_true.astype(
            np.uint8
        ),
        y_probability=(
            cross_fitted_probability.astype(
                np.float32
            )
        ),
        raw_y_probability=(
            raw_probability.astype(
                np.float32
            )
        ),
        calibration_method=np.asarray(
            selected_method
        ),
    )

    raw_curve = calibration_table(
        y_true,
        raw_probability,
    )
    calibrated_curve = (
        calibration_table(
            y_true,
            cross_fitted_probability,
        )
    )

    curve_rows = []

    for curve_name, rows in [
        ("raw", raw_curve),
        (
            "cross_fitted_calibrated",
            calibrated_curve,
        ),
    ]:
        for row in rows:
            curve_rows.append(
                {
                    "curve": curve_name,
                    **row,
                }
            )

    save_csv(
        output_dir
        / "calibration_curve.csv",
        curve_rows,
    )

    plot_roc(
        output_dir / "roc_curve.png",
        y_true,
        cross_fitted_probability,
        model_name,
    )
    plot_pr(
        output_dir / "pr_curve.png",
        y_true,
        cross_fitted_probability,
        model_name,
    )
    plot_threshold_f1(
        output_dir
        / "threshold_f1_curve.png",
        precision,
        recall,
        thresholds,
        best_threshold,
        best_f1,
        model_name,
    )
    plot_confusion(
        output_dir
        / "confusion_matrix.png",
        y_true,
        cross_fitted_probability,
        best_threshold,
        model_name,
    )
    plot_calibration_curve(
        output_dir
        / "calibration_curve.png",
        raw_curve,
        calibrated_curve,
        model_name,
        raw_metrics,
        calibrated_probability_metrics,
    )
    plot_method_comparison(
        output_dir
        / "calibration_metric_comparison.png",
        comparison_rows,
        model_name,
    )

    append_log(
        log_path,
        (
            f"Saved calibrated model: "
            f"{output_dir / 'model.joblib'}"
        ),
    )
    append_log(
        log_path,
        (
            f"Calibrated threshold="
            f"{best_threshold:.8f}, "
            f"F1={metrics['f1']:.8f}, "
            f"Brier={metrics['brier_score']:.8f}, "
            f"LogLoss={metrics['log_loss']:.8f}, "
            f"ECE={metrics['ece']:.8f}"
        ),
    )

    summary = {
        "model": model_name,
        "selected_method": selected_method,
        "raw_brier_score": (
            raw_metrics[
                "brier_score"
            ]
        ),
        "calibrated_brier_score": (
            calibrated_probability_metrics[
                "brier_score"
            ]
        ),
        "brier_improvement": (
            raw_metrics[
                "brier_score"
            ]
            - calibrated_probability_metrics[
                "brier_score"
            ]
        ),
        "raw_log_loss": (
            raw_metrics[
                "log_loss"
            ]
        ),
        "calibrated_log_loss": (
            calibrated_probability_metrics[
                "log_loss"
            ]
        ),
        "log_loss_improvement": (
            raw_metrics[
                "log_loss"
            ]
            - calibrated_probability_metrics[
                "log_loss"
            ]
        ),
        "raw_ece": raw_metrics["ece"],
        "calibrated_ece": (
            calibrated_probability_metrics[
                "ece"
            ]
        ),
        "ece_improvement": (
            raw_metrics["ece"]
            - calibrated_probability_metrics[
                "ece"
            ]
        ),
        "raw_roc_auc": (
            raw_metrics[
                "roc_auc"
            ]
        ),
        "calibrated_roc_auc": (
            calibrated_probability_metrics[
                "roc_auc"
            ]
        ),
        "raw_average_precision": (
            raw_metrics[
                "average_precision"
            ]
        ),
        "calibrated_average_precision": (
            calibrated_probability_metrics[
                "average_precision"
            ]
        ),
        "selected_threshold": (
            best_threshold
        ),
        "calibrated_f1": metrics["f1"],
        "calibrated_precision": (
            metrics["precision"]
        ),
        "calibrated_recall": (
            metrics["recall"]
        ),
    }

    del base_model
    del wrapped_model
    del final_calibrator
    del y_true
    del raw_probability
    del cross_fitted_probability
    gc.collect()

    return summary


# =============================================================================
# 7. 汇总
# =============================================================================
def plot_summary(
    path: Path,
    rows: Sequence[
        Dict[str, object]
    ],
) -> None:
    models = [
        str(row["model"])
        for row in rows
    ]
    x = np.arange(len(models))
    width = 0.36

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    pairs = [
        (
            "raw_brier_score",
            "calibrated_brier_score",
            "Brier score",
        ),
        (
            "raw_log_loss",
            "calibrated_log_loss",
            "Log loss",
        ),
        (
            "raw_ece",
            "calibrated_ece",
            "ECE",
        ),
    ]

    for axis, (
        raw_key,
        calibrated_key,
        title,
    ) in zip(axes, pairs):
        raw_values = [
            float(row[raw_key])
            for row in rows
        ]
        calibrated_values = [
            float(row[calibrated_key])
            for row in rows
        ]

        axis.bar(
            x - width / 2,
            raw_values,
            width=width,
            label="Raw",
        )
        axis.bar(
            x + width / 2,
            calibrated_values,
            width=width,
            label="Calibrated",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(
            models,
            rotation=20,
            ha="right",
        )
        axis.set_title(title)
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    axes[0].legend()
    figure.suptitle(
        "Probability calibration summary"
    )

    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# 8. 主流程
# =============================================================================
def main() -> None:
    copy_checkpoint_structure()

    summary_rows: List[
        Dict[str, object]
    ] = []

    for model_name in ENABLED_MODELS:
        print("\n" + "=" * 80)
        print(
            f"[CALIBRATE] {model_name}"
        )

        summary = calibrate_one_model(
            model_name
        )
        summary_rows.append(summary)

    save_csv(
        OUTPUT_CHECKPOINT_ROOT
        / "calibration_summary.csv",
        summary_rows,
    )

    plot_summary(
        OUTPUT_CHECKPOINT_ROOT
        / "calibration_summary.png",
        summary_rows,
    )

    save_json(
        OUTPUT_CHECKPOINT_ROOT
        / "calibration_config.json",
        {
            "source_checkpoint_root": str(
                SOURCE_CHECKPOINT_ROOT
            ),
            "output_checkpoint_root": str(
                OUTPUT_CHECKPOINT_ROOT
            ),
            "enabled_models": (
                ENABLED_MODELS
            ),
            "candidate_methods": (
                CANDIDATE_METHODS
            ),
            "selection_objective": (
                "minimum leave-one-date-out "
                "cross-fitted Brier score; "
                "log loss and ECE as tie breakers"
            ),
            "threshold_objective": (
                THRESHOLD_OBJECTIVE
            ),
            "max_calibration_fit_samples": (
                MAX_CALIBRATION_FIT_SAMPLES
            ),
            "calibration_bin_count": (
                CALIBRATION_BIN_COUNT
            ),
            "random_state": (
                RANDOM_STATE
            ),
        },
    )

    print("\n[DONE]")
    print(
        "Calibrated checkpoint root:",
        OUTPUT_CHECKPOINT_ROOT.resolve(),
    )
    print(
        "Switch prediction CHECKPOINT_ROOT "
        "between:"
    )
    print(
        "  raw        :",
        SOURCE_CHECKPOINT_ROOT,
    )
    print(
        "  calibrated :",
        OUTPUT_CHECKPOINT_ROOT,
    )


if __name__ == "__main__":
    main()