#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集中评估并比较广东省火灾分类模型。

本脚本输出两类结果：

一、交叉验证结果（推荐用于正式模型比较）
----------------------------------------
直接读取每个模型 checkpoint 中保存的：
    oof_predictions.npz
    threshold.json
    metrics.json

生成：
1. 多模型 ROC 曲线 + 配套 CSV
2. 多模型 PR 曲线 + 配套 CSV
3. 多模型 OOF 混淆矩阵总图 + 配套 CSV
4. 交叉验证总体指标比较图 + 配套 CSV
5. 最佳 F1 阈值比较图 + 配套 CSV

二、历史和预测地图统计
----------------------
历史目录默认：
    ./predictions

预测目录默认：
    ./predictions_from_pangu

历史数据存在标签，因此输出：
1. 每日、每模型分类指标
2. 每个模型在全部历史日期上的总体指标
3. 每日指标变化比较图
4. 历史地图总图的配套数据表
5. 历史模型比较图的配套数据表

预测数据没有标签，因此只输出：
1. 每日平均/最大概率
2. 每日预测火灾像元数量和比例
3. 预测期描述性统计比较图
4. 每个模型预测总图的配套数据表

重要说明
--------
- 预测数据没有真实标签，不能计算 Accuracy、F1、ROC AUC 等指标。
- 历史地图若由“使用全部历史日期重新训练的最终模型”产生，
  则其地图指标属于训练集或拟合后指标，不等同于独立测试结果。
- 正式比较优先使用 checkpoint OOF 结果。
- 每一张指标图都生成一个同名或明确对应的 CSV 表。
- 原始概率图和类别图的完整逐像元数据已经保存在 NPY 文件中；
  CSV 记录图中对应日期和模型的统计量，而不是重复展开数百万像元。

依赖
----
conda install -c conda-forge \
    numpy pandas matplotlib scikit-learn

运行
----
python evaluate_fire_models_and_export_tables.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
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


# =============================================================================
# 0. 配置
# =============================================================================


# 有 OOF checkpoint 的基础模型
BASE_MODEL_NAMES = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
]

# 地图输出中需要比较的模型，可包含 ensemble
PREDICTION_MODEL_NAMES = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
    "ensemble",
]

# 历史预测结果
# CHECKPOINT_ROOT = Path("./checkpoint")
# HISTORICAL_ROOT = Path("./workspace/火灾预测/historical_predictions")
# FORECAST_ROOT = Path("./workspace/火灾预测/pangu_predictions")
# OUTPUT_ROOT = Path("./workspace/训练指标可视化/evaluate_fire_models_and_export_tables")
CHECKPOINT_ROOT = Path("./checkpoint_calibration")
HISTORICAL_ROOT = Path("./workspace/火灾预测/historical_predictions_calibration")
FORECAST_ROOT = Path("./workspace/火灾预测/pangu_predictions_calibration")
OUTPUT_ROOT = Path("./workspace/训练指标可视化/evaluate_fire_models_and_export_tables_calibration")
HIST_START_DATE = "20251201"
HIST_END_DATE = "20251231"

# 盘古预测结果

FORECAST_START_DATE = "20261201"
FORECAST_END_DATE = "20261231"

# 历史标签
LABEL_DIR = Path("./data/processed/firms_conf_1km")
LABEL_PATTERN = "firms_conf_{date}.npy"
LABEL_CONFIDENCE_THRESHOLD = 0.0

# 广东掩膜
MASK_PATH = Path("./data/processed/guangdong_mask_1km.npy")

# 输出
TABLE_DIR_NAME = "tables"
FIGURE_DIR_NAME = "figures"
PAIRED_TABLE_DIR_NAME = "map_figure_tables"

DPI = 250

# ROC/PR 曲线数据点过多时，CSV 最多保留多少个点/模型。
MAX_CURVE_TABLE_POINTS = 5000


# =============================================================================
# 1. 参数与通用工具
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate multiple fire classification models and export paired tables."
    )

    parser.add_argument("--checkpoint-root", default=str(CHECKPOINT_ROOT))
    parser.add_argument("--historical-root", default=str(HISTORICAL_ROOT))
    parser.add_argument("--forecast-root", default=str(FORECAST_ROOT))
    parser.add_argument("--hist-start-date", default=HIST_START_DATE)
    parser.add_argument("--hist-end-date", default=HIST_END_DATE)
    parser.add_argument("--forecast-start-date", default=FORECAST_START_DATE)
    parser.add_argument("--forecast-end-date", default=FORECAST_END_DATE)
    parser.add_argument("--label-dir", default=str(LABEL_DIR))
    parser.add_argument("--label-pattern", default=LABEL_PATTERN)
    parser.add_argument("--label-threshold", type=float, default=LABEL_CONFIDENCE_THRESHOLD)
    parser.add_argument("--mask-path", default=str(MASK_PATH))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def parse_yyyymmdd(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d")


def date_range(start_date: str, end_date: str) -> List[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    if end < start:
        raise ValueError("结束日期不能早于开始日期。")
    dates: List[str] = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved table:", path)


def load_mask(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到广东掩膜：{path}")
    mask = np.load(path)
    if mask.ndim != 2:
        raise ValueError(f"广东掩膜应为二维，当前 shape={mask.shape}")
    return mask.astype(bool)


# =============================================================================
# 2. 文件路径和数组读取
# =============================================================================
def get_prediction_paths(root: Path, model_name: str, date_text: str) -> Tuple[Path, Path, Path]:
    model_dir = root / model_name
    if model_name == "ensemble":
        probability_path = model_dir / f"mean_probability_{date_text}.npy"
        binary_path = model_dir / f"majority_vote_{date_text}.npy"
    else:
        probability_path = model_dir / f"fire_probability_{date_text}.npy"
        binary_path = model_dir / f"fire_binary_{date_text}.npy"
    summary_path = model_dir / f"prediction_summary_{date_text}.json"
    return probability_path, binary_path, summary_path


def load_2d_array(path: Path, expected_shape: Tuple[int, int], kind: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到 {kind}：{path}")
    array = np.load(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{path} 应为二维，当前 shape={array.shape}")
    if array.shape != expected_shape:
        raise ValueError(f"{path} shape={array.shape}，期望 {expected_shape}")
    return np.asarray(array)


def load_probability(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    probability = load_2d_array(path, expected_shape, "概率图").astype(np.float32)
    probability = np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)
    probability = np.clip(probability, 0.0, 1.0)
    return probability


def load_binary(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    binary = load_2d_array(path, expected_shape, "类别图")
    binary = np.nan_to_num(binary, nan=0.0, posinf=0.0, neginf=0.0)
    return (binary > 0).astype(np.uint8)


def load_label_binary(label_dir: Path, label_pattern: str, date_text: str, expected_shape: Tuple[int, int], threshold: float) -> np.ndarray:
    path = label_dir / label_pattern.format(date=date_text)
    confidence = load_2d_array(path, expected_shape, "历史标签").astype(np.float32)
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    return (confidence > threshold).astype(np.uint8)


# =============================================================================
# 3. 指标计算
# =============================================================================
def safe_roc_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def safe_average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    if np.count_nonzero(y_true == 1) == 0:
        return float("nan")
    return float(average_precision_score(y_true, probability))


def compute_binary_metrics(y_true: np.ndarray, probability: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)
    probability = np.asarray(probability, dtype=np.float32)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "sample_count": int(y_true.size),
        "true_fire_pixels": int(np.count_nonzero(y_true == 1)),
        "true_no_fire_pixels": int(np.count_nonzero(y_true == 0)),
        "predicted_fire_pixels": int(np.count_nonzero(y_pred == 1)),
        "predicted_no_fire_pixels": int(np.count_nonzero(y_pred == 0)),
        "mean_probability": float(probability.mean()),
        "maximum_probability": float(probability.max()),
        "minimum_probability": float(probability.min()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, probability),
        "average_precision": safe_average_precision(y_true, probability),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def compute_forecast_statistics(probability: np.ndarray, binary: np.ndarray) -> Dict[str, object]:
    pixel_count = int(probability.size)
    fire_count = int(np.count_nonzero(binary == 1))
    return {
        "pixel_count": pixel_count,
        "predicted_fire_pixels": fire_count,
        "predicted_no_fire_pixels": int(pixel_count - fire_count),
        "predicted_fire_fraction": float(fire_count / pixel_count if pixel_count > 0 else 0.0),
        "mean_probability": float(probability.mean()),
        "median_probability": float(np.median(probability)),
        "maximum_probability": float(probability.max()),
        "minimum_probability": float(probability.min()),
        "probability_std": float(probability.std()),
        "probability_p90": float(np.percentile(probability, 90)),
        "probability_p95": float(np.percentile(probability, 95)),
        "probability_p99": float(np.percentile(probability, 99)),
    }


# =============================================================================
# 4. checkpoint OOF 结果
# =============================================================================
def load_checkpoint_oof(checkpoint_root: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, object]]:
    model_dir = checkpoint_root / model_name
    oof_path = model_dir / "oof_predictions.npz"
    threshold_path = model_dir / "threshold.json"
    metrics_path = model_dir / "metrics.json"
    if not oof_path.exists():
        raise FileNotFoundError(f"找不到 OOF 预测：{oof_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"找不到模型阈值：{threshold_path}")

    with np.load(oof_path) as payload:
        y_true = np.asarray(payload["y_true"], dtype=np.uint8)
        probability = np.asarray(payload["y_probability"], dtype=np.float32)

    threshold = float(load_json(threshold_path)["threshold"])
    stored_metrics = load_json(metrics_path) if metrics_path.exists() else {}
    return y_true, probability, threshold, stored_metrics


def sample_curve_indices(point_count: int, max_points: int) -> np.ndarray:
    if point_count <= max_points:
        return np.arange(point_count)
    return np.unique(np.linspace(0, point_count - 1, max_points, dtype=int))


def evaluate_checkpoint_oof(checkpoint_root: Path, table_dir: Path, figure_dir: Path, dpi: int) -> None:
    overall_rows, threshold_rows, roc_rows, pr_rows, confusion_rows = [], [], [], [], []
    oof_payloads: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    for model_name in BASE_MODEL_NAMES:
        y_true, probability, threshold, stored_metrics = load_checkpoint_oof(checkpoint_root, model_name)
        y_pred = (probability >= threshold).astype(np.uint8)
        computed = compute_binary_metrics(y_true, probability, y_pred)
        computed["model"] = model_name
        computed["threshold"] = threshold
        for key in ["validation_day_count", "validation_days"]:
            if key in stored_metrics:
                value = stored_metrics[key]
                if isinstance(value, list):
                    value = "|".join(map(str, value))
                computed[key] = value
        overall_rows.append(computed)
        threshold_rows.append({"model": model_name, "threshold": threshold, "oof_f1": computed["f1"]})

        fpr, tpr, roc_thresholds = roc_curve(y_true, probability)
        for index in sample_curve_indices(len(fpr), MAX_CURVE_TABLE_POINTS):
            roc_rows.append({
                "model": model_name,
                "false_positive_rate": float(fpr[index]),
                "true_positive_rate": float(tpr[index]),
                "threshold": float(roc_thresholds[index]),
                "roc_auc": computed["roc_auc"],
            })

        precision, recall, pr_thresholds = precision_recall_curve(y_true, probability)
        for index in sample_curve_indices(len(precision), MAX_CURVE_TABLE_POINTS):
            threshold_value = float(pr_thresholds[index]) if index < len(pr_thresholds) else float("nan")
            pr_rows.append({
                "model": model_name,
                "recall": float(recall[index]),
                "precision": float(precision[index]),
                "threshold": threshold_value,
                "average_precision": computed["average_precision"],
            })

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        for true_class in [0, 1]:
            for predicted_class in [0, 1]:
                confusion_rows.append({
                    "model": model_name,
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "count": int(cm[true_class, predicted_class]),
                })

        oof_payloads[model_name] = (y_true, probability, threshold)

    overall_df = pd.DataFrame(overall_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    roc_df = pd.DataFrame(roc_rows)
    pr_df = pd.DataFrame(pr_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    overall_table_path = table_dir / "checkpoint_cv_metrics.csv"
    threshold_table_path = table_dir / "selected_thresholds.csv"
    roc_table_path = table_dir / "combined_oof_roc_curve_data.csv"
    pr_table_path = table_dir / "combined_oof_pr_curve_data.csv"
    confusion_table_path = table_dir / "combined_oof_confusion_matrix_data.csv"

    save_dataframe(overall_df, overall_table_path)
    save_dataframe(threshold_df, threshold_table_path)
    save_dataframe(roc_df, roc_table_path)
    save_dataframe(pr_df, pr_table_path)
    save_dataframe(confusion_df, confusion_table_path)

    # 中文模型名称
    MODEL_DISPLAY_NAMES = {
        "logistic_regression": "逻辑回归",
        "sgd_logistic": "随机梯度下降逻辑分类器",
        "random_forest": "随机森林",
        "extra_trees": "极端随机树",
    }
    # 英文模型名称
    # MODEL_DISPLAY_NAMES = {
    #     "logistic_regression": "Logistic Regression",
    #     "sgd_logistic": "SGD Logistic",
    #     "random_forest": "Random Forest",
    #     "extra_trees": "Extra Trees",
    # }

    # 设置中文字体，并保证负号正常显示
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axis = plt.subplots(
        figsize=(7, 6),
        constrained_layout=True,
    )

    for model_name in BASE_MODEL_NAMES:
        y_true, probability, threshold = oof_payloads[model_name]

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            probability,
        )
        auc = roc_auc_score(
            y_true,
            probability,
        )

        chinese_model_name = MODEL_DISPLAY_NAMES.get(
            model_name,
            model_name,
        )

        # 绘制受试者工作特征曲线
        line, = axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2.5,
            label=(
                f"{chinese_model_name}"
                f"（AUC={auc:.3f}）"
            ),
        )

        # 根据最大 F1 分数对应的参考概率生成分类结果
        predicted_label = (
            probability >= threshold
        ).astype(np.uint8)

        true_negative, false_positive, false_negative, true_positive = (
            confusion_matrix(
                y_true,
                predicted_label,
                labels=[0, 1],
            ).ravel()
        )

        # 计算参考概率对应的假正率
        threshold_false_positive_rate = (
            false_positive
            / (false_positive + true_negative)
            if (false_positive + true_negative) > 0
            else 0.0
        )

        # 计算参考概率对应的真正率
        threshold_true_positive_rate = (
            true_positive
            / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        # 标记最大 F1 分数对应的工作点
        axis.scatter(
            threshold_false_positive_rate,
            threshold_true_positive_rate,
            s=110,
            color=line.get_color(),
            edgecolors="black",
            linewidths=1.1,
            zorder=5,
        )

        # axis.annotate(
        #     f"参考概率={threshold:.4f}",
        #     xy=(
        #         threshold_false_positive_rate,
        #         threshold_true_positive_rate,
        #     ),
        #     xytext=(8, 8),
        #     textcoords="offset points",
        #     fontsize=12,
        #     fontweight="bold",
        #     color=line.get_color(),
        # )

    # 随机分类参考线
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2.0,
        label="随机分类参考线",
    )
    # axis.plot(
    #     [0, 1],
    #     [0, 1],
    #     linestyle="--",
    #     linewidth=2.0,
    #     label="Random classification\nreference line",
    # )

    axis.set_xlabel(
        "假阳性率",
        fontsize=20,
        labelpad=10,
    )
    axis.set_ylabel(
        "真阳性率",
        fontsize=20,
        labelpad=10,
    )
    # axis.set_xlabel(
    #     "FPR",
    #     fontsize=20,
    #     labelpad=10,
    # )
    # axis.set_ylabel(
    #     "TPR",
    #     fontsize=20,
    #     labelpad=10,
    # )
    # axis.set_title(
    #     "按日期交叉验证的受试者工作特征曲线比较",
    #     fontsize=20,
    #     fontweight="bold",
    #     pad=16,
    # )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=14,
    )

    axis.grid(
        alpha=0.25,
        linewidth=1.0,
    )

    axis.legend(
        loc="upper left",
        fontsize=13,
        frameon=False,
        framealpha=0.95,
    )

    path = figure_dir / "combined_oof_roc_curve.png"
    pdf_path = figure_dir / "combined_oof_roc_curve.pdf"

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    print("已保存图像：", path, "；对应数据表：", roc_table_path,)

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for model_name in BASE_MODEL_NAMES:
        y_true, probability, _ = oof_payloads[model_name]
        precision, recall, _ = precision_recall_curve(y_true, probability)
        ap = average_precision_score(y_true, probability)
        axis.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Date-wise cross-validation PR comparison")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    path = figure_dir / "combined_oof_pr_curve.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print("Saved figure:", path, "| paired table:", pr_table_path)

    n_models = len(BASE_MODEL_NAMES)
    ncols = 2
    nrows = math.ceil(n_models / ncols)
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4.8 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    image = None
    for index, model_name in enumerate(BASE_MODEL_NAMES):
        axis = axes[index]
        y_true, probability, threshold = oof_payloads[model_name]
        y_pred = (probability >= threshold).astype(np.uint8)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        image = axis.imshow(cm)
        for row in range(2):
            for col in range(2):
                axis.text(col, row, str(int(cm[row, col])), ha="center", va="center")
        axis.set_xticks([0, 1]); axis.set_yticks([0, 1])
        axis.set_xticklabels(["No fire", "Fire"]); axis.set_yticklabels(["No fire", "Fire"])
        axis.set_xlabel("Predicted class"); axis.set_ylabel("True class")
        axis.set_title(f"{model_name}\nthreshold={threshold:.4f}")
    for index in range(len(BASE_MODEL_NAMES), len(axes)):
        axes[index].axis("off")
    figure.suptitle("OOF confusion matrices")
    if image is not None:
        figure.colorbar(image, ax=axes[:len(BASE_MODEL_NAMES)].tolist(), fraction=0.025, pad=0.02)
    path = figure_dir / "combined_oof_confusion_matrices.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print("Saved figure:", path, "| paired table:", confusion_table_path)

    metric_columns = ["f1", "precision", "recall", "specificity", "roc_auc", "average_precision", "balanced_accuracy", "mcc"]
    x = np.arange(len(BASE_MODEL_NAMES))
    bar_width = 0.09
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for metric_index, metric in enumerate(metric_columns):
        values = [float(overall_df.loc[overall_df["model"] == model, metric].iloc[0]) for model in BASE_MODEL_NAMES]
        offset = (metric_index - (len(metric_columns) - 1) / 2) * bar_width
        axis.bar(x + offset, values, width=bar_width, label=metric)
    axis.set_xticks(x)
    axis.set_xticklabels(BASE_MODEL_NAMES, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Metric value")
    axis.set_title("Date-wise cross-validation metric comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    path = figure_dir / "checkpoint_cv_metrics_comparison.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print("Saved figure:", path, "| paired table:", overall_table_path)

    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    axis.bar(threshold_df["model"], threshold_df["threshold"])
    axis.set_ylabel("Selected probability threshold")
    axis.set_title("Threshold selected by maximum OOF F1")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    for index, row in threshold_df.iterrows():
        axis.text(index, float(row["threshold"]), f"{float(row['threshold']):.3f}\nF1={float(row['oof_f1']):.3f}", ha="center", va="bottom", fontsize=8)
    path = figure_dir / "selected_thresholds_comparison.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print("Saved figure:", path, "| paired table:", threshold_table_path)


# =============================================================================
# 5. 历史地图指标
# =============================================================================
def evaluate_historical_maps(historical_root: Path, dates: Sequence[str], label_dir: Path, label_pattern: str, label_threshold: float, mask: np.ndarray, table_dir: Path, figure_dir: Path, paired_table_dir: Path, dpi: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: List[Dict[str, object]] = []
    overall_rows: List[Dict[str, object]] = []

    model_all_true = {model: [] for model in PREDICTION_MODEL_NAMES}
    model_all_probability = {model: [] for model in PREDICTION_MODEL_NAMES}
    model_all_pred = {model: [] for model in PREDICTION_MODEL_NAMES}

    for date_text in dates:
        label_map = load_label_binary(label_dir, label_pattern, date_text, mask.shape, label_threshold)
        y_true = label_map[mask]

        for model_name in PREDICTION_MODEL_NAMES:
            probability_path, binary_path, summary_path = get_prediction_paths(historical_root, model_name, date_text)
            if not probability_path.exists() or not binary_path.exists():
                print("[MISSING historical]", model_name, date_text)
                continue

            probability_map = load_probability(probability_path, mask.shape)
            binary_map = load_binary(binary_path, mask.shape)
            probability = probability_map[mask]
            y_pred = binary_map[mask]

            metrics = compute_binary_metrics(y_true, probability, y_pred)
            metrics["model"] = model_name
            metrics["date"] = date_text

            if summary_path.exists():
                summary = load_json(summary_path)
                metrics["threshold"] = float(summary["threshold"]) if "threshold" in summary else float("nan")
            else:
                metrics["threshold"] = float("nan")

            daily_rows.append(metrics)
            model_all_true[model_name].append(np.asarray(y_true, dtype=np.uint8))
            model_all_probability[model_name].append(np.asarray(probability, dtype=np.float32))
            model_all_pred[model_name].append(np.asarray(y_pred, dtype=np.uint8))

    for model_name in PREDICTION_MODEL_NAMES:
        if not model_all_true[model_name]:
            continue
        y_true = np.concatenate(model_all_true[model_name])
        probability = np.concatenate(model_all_probability[model_name])
        y_pred = np.concatenate(model_all_pred[model_name])
        metrics = compute_binary_metrics(y_true, probability, y_pred)
        metrics["model"] = model_name
        metrics["date_count"] = len(model_all_true[model_name])
        overall_rows.append(metrics)

    daily_df = pd.DataFrame(daily_rows)
    overall_df = pd.DataFrame(overall_rows)

    daily_table_path = table_dir / "historical_daily_metrics.csv"
    overall_table_path = table_dir / "historical_map_overall_metrics.csv"
    save_dataframe(daily_df, daily_table_path)
    save_dataframe(overall_df, overall_table_path)

    for model_name in PREDICTION_MODEL_NAMES:
        subset = daily_df[daily_df["model"] == model_name].copy()
        if not subset.empty:
            save_dataframe(subset, paired_table_dir / f"{model_name}_historical_3x7_data.csv")

    probability_columns = ["date", "model", "mean_probability", "maximum_probability", "minimum_probability", "true_fire_pixels", "predicted_fire_pixels", "roc_auc", "average_precision"]
    binary_columns = ["date", "model", "true_fire_pixels", "predicted_fire_pixels", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "tn", "fp", "fn", "tp"]

    save_dataframe(daily_df[probability_columns], paired_table_dir / "historical_compare_probability_7rows_data.csv")
    save_dataframe(daily_df[binary_columns], paired_table_dir / "historical_compare_binary_7rows_data.csv")

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 13), sharex=True, constrained_layout=True)
    for axis, metric in zip(axes, ["f1", "precision", "recall"]):
        for model_name in PREDICTION_MODEL_NAMES:
            subset = daily_df[daily_df["model"] == model_name].sort_values("date")
            if subset.empty:
                continue
            axis.plot(subset["date"], subset[metric], marker="o", label=model_name)
        axis.set_ylabel(metric)
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
    axes[0].set_title("Historical daily classification metrics")
    axes[-1].set_xlabel("Date")
    axes[-1].tick_params(axis="x", rotation=30)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    path = figure_dir / "historical_daily_metrics_comparison.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print("Saved figure:", path, "| paired table:", daily_table_path)

    metric_columns = ["f1", "precision", "recall", "specificity", "roc_auc", "average_precision", "balanced_accuracy"]
    model_order = overall_df["model"].tolist()
    x = np.arange(len(model_order))
    width = 0.11
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for metric_index, metric in enumerate(metric_columns):
        values = overall_df[metric].astype(float)
        offset = (metric_index - (len(metric_columns) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=metric)
    axis.set_xticks(x)
    axis.set_xticklabels(model_order, rotation=15, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Metric value")
    axis.set_title("Historical map-output metric comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    path = figure_dir / "historical_map_overall_metrics_comparison.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print("Saved figure:", path, "| paired table:", overall_table_path)

    return daily_df, overall_df


# =============================================================================
# 6. 预测地图描述性统计
# =============================================================================
def evaluate_forecast_maps(forecast_root: Path, dates: Sequence[str], mask: np.ndarray, table_dir: Path, figure_dir: Path, paired_table_dir: Path, dpi: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for date_text in dates:
        for model_name in PREDICTION_MODEL_NAMES:
            probability_path, binary_path, summary_path = get_prediction_paths(forecast_root, model_name, date_text)
            if not probability_path.exists() or not binary_path.exists():
                print("[MISSING forecast]", model_name, date_text)
                continue

            probability_map = load_probability(probability_path, mask.shape)
            binary_map = load_binary(binary_path, mask.shape)
            probability = probability_map[mask]
            binary = binary_map[mask]

            stats = compute_forecast_statistics(probability, binary)
            stats["model"] = model_name
            stats["date"] = date_text

            if summary_path.exists():
                summary = load_json(summary_path)
                stats["threshold"] = float(summary["threshold"]) if "threshold" in summary else float("nan")
            else:
                stats["threshold"] = float("nan")

            rows.append(stats)

    forecast_df = pd.DataFrame(rows)
    forecast_table_path = table_dir / "forecast_daily_summary.csv"
    save_dataframe(forecast_df, forecast_table_path)

    for model_name in PREDICTION_MODEL_NAMES:
        subset = forecast_df[forecast_df["model"] == model_name].copy()
        if not subset.empty:
            save_dataframe(subset, paired_table_dir / f"{model_name}_forecast_2x7_data.csv")

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for model_name in PREDICTION_MODEL_NAMES:
        subset = forecast_df[forecast_df["model"] == model_name].sort_values("date")
        if subset.empty:
            continue
        axes[0].plot(subset["date"], subset["mean_probability"], marker="o", label=model_name)
        axes[1].plot(subset["date"], subset["predicted_fire_pixels"], marker="o", label=model_name)

    axes[0].set_ylabel("Mean predicted probability")
    axes[0].set_title("Forecast descriptive comparison")
    axes[0].grid(alpha=0.25)
    axes[1].set_ylabel("Predicted fire pixels")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.25)
    axes[1].tick_params(axis="x", rotation=30)
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))

    path = figure_dir / "forecast_daily_summary_comparison.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print("Saved figure:", path, "| paired table:", forecast_table_path)

    return forecast_df


# =============================================================================
# 7. 输出索引
# =============================================================================
def write_output_index(output_root: Path, table_dir: Path, figure_dir: Path, paired_table_dir: Path) -> None:
    rows: List[Dict[str, str]] = []
    figure_to_table = {
        "combined_oof_roc_curve.png": "combined_oof_roc_curve_data.csv",
        "combined_oof_pr_curve.png": "combined_oof_pr_curve_data.csv",
        "combined_oof_confusion_matrices.png": "combined_oof_confusion_matrix_data.csv",
        "checkpoint_cv_metrics_comparison.png": "checkpoint_cv_metrics.csv",
        "selected_thresholds_comparison.png": "selected_thresholds.csv",
        "historical_daily_metrics_comparison.png": "historical_daily_metrics.csv",
        "historical_map_overall_metrics_comparison.png": "historical_map_overall_metrics.csv",
        "forecast_daily_summary_comparison.png": "forecast_daily_summary.csv",
    }

    for figure_name, table_name in figure_to_table.items():
        rows.append({
            "figure": str(figure_dir / figure_name),
            "paired_table": str(table_dir / table_name),
            "purpose": figure_name.replace(".png", "").replace("_", " "),
        })

    index_df = pd.DataFrame(rows)
    save_dataframe(index_df, output_root / "figure_table_index.csv")

    readme = f"""Evaluation output
=================

Figures:
    {figure_dir}

Metric tables:
    {table_dir}

Tables paired with map figures from visualize_fire_prediction_grids.py:
    {paired_table_dir}

The exact figure-table pairing is recorded in:
    {output_root / 'figure_table_index.csv'}

Interpretation:
- checkpoint OOF figures/tables are the preferred cross-validation comparison.
- historical map metrics evaluate saved historical map outputs.
- forecast tables contain descriptive statistics only because no labels exist.
"""
    with open(output_root / "README.txt", "w", encoding="utf-8") as file:
        file.write(readme)


# =============================================================================
# 8. 主流程
# =============================================================================
def main() -> None:
    args = parse_args()

    checkpoint_root = Path(args.checkpoint_root)
    historical_root = Path(args.historical_root)
    forecast_root = Path(args.forecast_root)
    label_dir = Path(args.label_dir)
    mask_path = Path(args.mask_path)
    output_root = Path(args.output_root)

    table_dir = output_root / TABLE_DIR_NAME
    figure_dir = output_root / FIGURE_DIR_NAME
    paired_table_dir = output_root / PAIRED_TABLE_DIR_NAME

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paired_table_dir.mkdir(parents=True, exist_ok=True)

    mask = load_mask(mask_path)
    hist_dates = date_range(args.hist_start_date, args.hist_end_date)
    forecast_dates = date_range(args.forecast_start_date, args.forecast_end_date)

    print("\n[1/3] Evaluate checkpoint OOF results")
    evaluate_checkpoint_oof(checkpoint_root=checkpoint_root, table_dir=table_dir, figure_dir=figure_dir, dpi=args.dpi)
    exit()
    print("\n[2/3] Evaluate historical map outputs")
    evaluate_historical_maps(
        historical_root=historical_root,
        dates=hist_dates,
        label_dir=label_dir,
        label_pattern=args.label_pattern,
        label_threshold=args.label_threshold,
        mask=mask,
        table_dir=table_dir,
        figure_dir=figure_dir,
        paired_table_dir=paired_table_dir,
        dpi=args.dpi,
    )

    print("\n[3/3] Summarize forecast map outputs")
    evaluate_forecast_maps(
        forecast_root=forecast_root,
        dates=forecast_dates,
        mask=mask,
        table_dir=table_dir,
        figure_dir=figure_dir,
        paired_table_dir=paired_table_dir,
        dpi=args.dpi,
    )

    write_output_index(output_root=output_root, table_dir=table_dir, figure_dir=figure_dir, paired_table_dir=paired_table_dir)

    print("\n[DONE]")
    print("Output root:", output_root.resolve())
    print("Figures:", figure_dir.resolve())
    print("Metric tables:", table_dir.resolve())
    print("Map-figure paired tables:", paired_table_dir.resolve())


if __name__ == "__main__":
    main()