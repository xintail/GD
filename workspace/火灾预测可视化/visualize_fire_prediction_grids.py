#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成火灾预测月尺度可视化比较图。

输出内容
--------
默认生成 2 张历史模型比较图：

1) 月平均概率比较图（2 行 × 模型数 列）：
   - 上排：各模型 2025 年 12 月的月平均概率空间分布；
   - 下排：各模型全省平均概率的逐日曲线（12 月 1—31 日）。

2) 月内火灾判定日数比较图（2 行 × (模型数 + 1) 列）：
   - 上排：各模型月内判定为火灾的天数，最后一列为真实火点出现天数；
   - 下排：各模型与真实标签的逐日火灾像元数曲线。

说明
----
- 本脚本默认不生成每个模型的 31 列逐日总图；如需生成，可将
  GENERATE_PER_MODEL_FIGURES 设为 True。
- 历史预测目录默认为 ./workspace/火灾预测/historical_predictions，
  可通过 --historical-root 切换为校准版本目录。
- 标签目录默认为 ./data/processed/firms_conf_1km。
- 支持 ensemble 目录；若 model_name == "ensemble"：
      概率图读取 mean_probability_YYYYMMDD.npy
      类别图读取 majority_vote_YYYYMMDD.npy
  否则读取：
      fire_probability_YYYYMMDD.npy
      fire_binary_YYYYMMDD.npy

依赖
----
conda install -c conda-forge numpy matplotlib geopandas shapely rasterio

运行
----
python visualize_fire_prediction_grids.py
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator


# =============================================================================
# 0. 默认配置
# =============================================================================
HISTORICAL_ROOT = Path("./workspace/火灾预测/historical_predictions")
FORECAST_ROOT = Path("./workspace/火灾预测/pangu_predictions")
OUTPUT_ROOT = Path("./workspace/火灾预测可视化/visualize_fire_prediction_grids")
# HISTORICAL_ROOT = Path("./workspace/火灾预测/historical_predictions_calibration")
# FORECAST_ROOT = Path("./workspace/火灾预测/pangu_predictions_calibration")
# OUTPUT_ROOT = Path("./workspace/火灾预测可视化/visualize_fire_prediction_grids_calibration")
# 这里定义要分别输出历史/预测总图的模型列表。
PER_MODEL_NAMES = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
    "ensemble",
]

# 这里定义要用于“模型比较图”的模型列表。
# 由于比较图是用于比较模型，通常可包含 ensemble，也可只比较基础模型。
COMPARE_MODEL_NAMES = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
    # "ensemble",
]

# 历史 12 月整月
HIST_START_DATE = "20251201"
HIST_END_DATE = "20251231"

# 预测 12 月整月
FORECAST_START_DATE = "20261201"
FORECAST_END_DATE = "20261231"

# 是否生成每个模型的历史/预测总图。
# 本月报告只使用模型比较图，且 31 天逐日总图可读性较差，默认关闭。
GENERATE_PER_MODEL_FIGURES = False

LABEL_DIR = Path("./data/processed/firms_conf_1km")
LABEL_PATTERN = "firms_conf_{date}.npy"
LABEL_CONFIDENCE_THRESHOLD = 0.0

GRID_METADATA_PATH = Path("./data/processed/target_grid_metadata.json")
GUANGDONG_MASK_PATH = Path("./data/processed/guangdong_mask_1km.npy")
BOUNDARY_SHP = Path("./data/boundary/gd_boundary.shp")


SAVE_PDF = False
DPI = 220

# 概率图中是否隐藏 0 值背景。
MASK_ZERO_PROBABILITY = True

# 二值图颜色：0=白色，1=黑色
BINARY_CMAP = ListedColormap(["white", "black"])

# 中文字体设置。只修改文字显示方式，不改变原有数据读取与绘图逻辑。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# historical_comparison 中使用的模型中文名称。
# COMPARE_MODEL_DISPLAY_NAMES = {
#     "logistic_regression": "逻辑回归",
#     "sgd_logistic": "随机梯度下降逻辑分类器",
#     "random_forest": "随机森林",
#     "extra_trees": "极端随机树",
#     "ensemble": "集成模型",
# }
# historical_comparison 中使用的模型英文名称。
COMPARE_MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "sgd_logistic": "SGD Logistic",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "ensemble": "Ensemble",
}

# 模型在逐日曲线中的统一颜色。
MODEL_COLORS = {
    "logistic_regression": "#1f77b4",
    "sgd_logistic": "#ff7f0e",
    "random_forest": "#2ca02c",
    "extra_trees": "#d62728",
    "ensemble": "#9467bd",
}

# historical_comparison 中的字号。
COMPARE_TITLE_FONTSIZE = 25
COMPARE_DATE_FONTSIZE = 22
COMPARE_TICK_FONTSIZE = 18
COMPARE_COLORBAR_LABEL_FONTSIZE = 20
COMPARE_COLORBAR_TICK_FONTSIZE = 18


# =============================================================================
# 1. 参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-model and cross-model grid figures for fire prediction."
    )

    parser.add_argument("--historical-root", default=str(HISTORICAL_ROOT))
    parser.add_argument("--forecast-root", default=str(FORECAST_ROOT))

    parser.add_argument(
        "--hist-start-date",
        default=HIST_START_DATE,
        help="Historical start date in YYYYMMDD",
    )
    parser.add_argument(
        "--hist-end-date",
        default=HIST_END_DATE,
        help="Historical end date in YYYYMMDD",
    )
    parser.add_argument(
        "--forecast-start-date",
        default=FORECAST_START_DATE,
        help="Forecast start date in YYYYMMDD",
    )
    parser.add_argument(
        "--forecast-end-date",
        default=FORECAST_END_DATE,
        help="Forecast end date in YYYYMMDD",
    )

    parser.add_argument(
        "--label-dir",
        default=str(LABEL_DIR),
    )
    parser.add_argument(
        "--label-pattern",
        default=LABEL_PATTERN,
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=LABEL_CONFIDENCE_THRESHOLD,
    )

    parser.add_argument(
        "--grid-metadata",
        default=str(GRID_METADATA_PATH),
    )
    parser.add_argument(
        "--boundary-shp",
        default=str(BOUNDARY_SHP),
    )
    parser.add_argument(
        "--mask",
        default=str(GUANGDONG_MASK_PATH),
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        default=SAVE_PDF,
    )
    parser.add_argument(
        "--mask-zero-probability",
        action="store_true",
        default=MASK_ZERO_PROBABILITY,
    )

    return parser.parse_args()


# =============================================================================
# 2. 日期与元数据
# =============================================================================
def parse_yyyymmdd(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d")


def date_range(start_date: str, end_date: str) -> List[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)

    if end < start:
        raise ValueError("结束日期不能早于开始日期。")

    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def format_date_chinese(date_text: str) -> str:
    """将 YYYYMMDD 日期转换为中文日期。"""
    date_value = parse_yyyymmdd(date_text)
    return f"{date_value.year}.{date_value.month}.{date_value.day}"


def require_dates(dates: Sequence[str], label: str) -> None:
    if not dates:
        raise ValueError(f"{label}为空，请检查起止日期。")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_extent_and_shape(
    grid_metadata_path: Path,
) -> Tuple[Tuple[float, float, float, float], Tuple[int, int]]:
    if not grid_metadata_path.exists():
        raise FileNotFoundError(
            f"找不到网格元数据：{grid_metadata_path}"
        )

    metadata = load_json(grid_metadata_path)

    bounds = metadata["bounds"]
    extent = (
        float(bounds["left"]),
        float(bounds["right"]),
        float(bounds["bottom"]),
        float(bounds["top"]),
    )
    shape = (
        int(metadata["height"]),
        int(metadata["width"]),
    )

    return extent, shape


# =============================================================================
# 3. 坐标格式
# =============================================================================
def format_longitude(value, position=None):
    if value > 180:
        value -= 360
    if value < 0:
        return f"{abs(value):.1f}°W"
    return f"{value:.1f}°E"


def format_latitude(value, position=None):
    if value < 0:
        return f"{abs(value):.1f}°S"
    return f"{value:.1f}°N"


def make_ticks(minimum: float, maximum: float, count: int = 4):
    return np.linspace(minimum, maximum, count)


# =============================================================================
# 4. 路径规则
# =============================================================================
def get_prediction_paths(
    root: Path,
    model_name: str,
    date_text: str,
) -> Tuple[Path, Path]:
    model_dir = root / model_name

    if model_name == "ensemble":
        probability_path = model_dir / f"mean_probability_{date_text}.npy"
        binary_path = model_dir / f"majority_vote_{date_text}.npy"
    else:
        probability_path = model_dir / f"fire_probability_{date_text}.npy"
        binary_path = model_dir / f"fire_binary_{date_text}.npy"

    return probability_path, binary_path


def get_label_path(
    label_dir: Path,
    label_pattern: str,
    date_text: str,
) -> Path:
    return label_dir / label_pattern.format(date=date_text)


# =============================================================================
# 5. 数据读取
# =============================================================================
def load_probability_map(
    path: Path,
    expected_shape: Tuple[int, int],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到概率图：{path}")

    array = np.load(path)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"{path} 应为二维数组，当前为 {array.shape}")

    if array.shape != expected_shape:
        raise ValueError(
            f"{path} shape={array.shape}，期望 {expected_shape}"
        )

    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array[array < 0] = 0.0
    array[array > 1] = 1.0

    return array


def load_binary_map(
    path: Path,
    expected_shape: Tuple[int, int],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到类别图：{path}")

    array = np.load(path)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"{path} 应为二维数组，当前为 {array.shape}")

    if array.shape != expected_shape:
        raise ValueError(
            f"{path} shape={array.shape}，期望 {expected_shape}"
        )

    array = np.asarray(array)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = (array > 0).astype(np.uint8)

    return array


def load_label_binary(
    path: Path,
    expected_shape: Tuple[int, int],
    threshold: float,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到标签文件：{path}")

    array = np.load(path)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 2:
        raise ValueError(f"{path} 应为二维数组，当前为 {array.shape}")

    if array.shape != expected_shape:
        raise ValueError(
            f"{path} shape={array.shape}，期望 {expected_shape}"
        )

    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    return (array > threshold).astype(np.uint8)


# =============================================================================
# 6. 绘图辅助
# =============================================================================
def load_boundary(boundary_shp: Path) -> gpd.GeoDataFrame:
    if not boundary_shp.exists():
        raise FileNotFoundError(f"找不到广东边界：{boundary_shp}")

    boundary = gpd.read_file(boundary_shp)

    if boundary.empty:
        raise ValueError("广东边界文件为空。")
    if boundary.crs is None:
        raise ValueError("广东边界缺少 CRS，请检查 gd_boundary.prj。")

    return boundary.to_crs("EPSG:4326")


def draw_map_common(
    axis,
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    show_xticks: bool,
    show_yticks: bool,
):
    left, right, bottom, top = extent

    boundary.boundary.plot(
        ax=axis,
        linewidth=0.8,
    )

    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)

    if show_xticks:
        axis.set_xticks(make_ticks(left, right, count=4))
        axis.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    else:
        axis.set_xticks([])

    if show_yticks:
        axis.set_yticks(make_ticks(bottom, top, count=4))
        axis.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    else:
        axis.set_yticks([])

    axis.set_xlabel("")
    axis.set_ylabel("")


def plot_probability(
    axis,
    probability: np.ndarray,
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    show_xticks: bool,
    show_yticks: bool,
    mask_zero_probability: bool,
    vmin: float,
    vmax: float,
):
    display = probability
    if mask_zero_probability:
        display = np.ma.masked_where(probability <= 0, probability)

    image = axis.imshow(
        display,
        extent=extent,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )

    draw_map_common(
        axis,
        boundary,
        extent,
        show_xticks=show_xticks,
        show_yticks=show_yticks,
    )

    return image


def plot_binary(
    axis,
    binary_map: np.ndarray,
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    show_xticks: bool,
    show_yticks: bool,
):
    image = axis.imshow(
        binary_map,
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=1,
        cmap=BINARY_CMAP,
        interpolation="nearest",
        aspect="auto",
    )

    draw_map_common(
        axis,
        boundary,
        extent,
        show_xticks=show_xticks,
        show_yticks=show_yticks,
    )

    return image


# =============================================================================
# 7. 单模型总图
# =============================================================================
def visualize_per_model_historical(
    model_name: str,
    dates: Sequence[str],
    root: Path,
    label_dir: Path,
    label_pattern: str,
    label_threshold: float,
    expected_shape: Tuple[int, int],
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    mask_zero_probability: bool,
) -> None:
    figure, axes = plt.subplots(
        nrows=3,
        ncols=7,
        figsize=(28, 11),
        constrained_layout=True,
    )

    probability_image_for_cbar = None
    vmax = []
    vmin = []
    for _, date_text in enumerate(dates):
        probability_path, binary_path = get_prediction_paths(
            root,
            model_name,
            date_text,
        )
        label_path = get_label_path(
            label_dir,
            label_pattern,
            date_text,
        )

        probability = load_probability_map(probability_path, expected_shape)
        vmax.append(probability.max())
        vmin.append(probability.min())
    vmax = max(vmax)
    vmin = min(vmin)

    for col, date_text in enumerate(dates):
        probability_path, binary_path = get_prediction_paths(
            root,
            model_name,
            date_text,
        )
        label_path = get_label_path(
            label_dir,
            label_pattern,
            date_text,
        )

        probability = load_probability_map(probability_path, expected_shape)
        predicted_binary = load_binary_map(binary_path, expected_shape)
        label_binary = load_label_binary(label_path, expected_shape, label_threshold)

        # 第一行：概率图
        ax = axes[0, col]
        probability_image_for_cbar = plot_probability(
            ax,
            probability,
            boundary,
            extent,
            show_xticks=False,
            show_yticks=(col == 0),
            mask_zero_probability=mask_zero_probability,
            vmin=vmin, 
            vmax=vmax,
        )
        ax.set_title(date_text, fontsize=11)
        if col == 0:
            ax.text(
                -0.08, 0.5, "Probability",
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=12, fontweight="bold",
            )

        # 第二行：类别图
        ax = axes[1, col]
        plot_binary(
            ax,
            predicted_binary,
            boundary,
            extent,
            show_xticks=False,
            show_yticks=(col == 0),
        )
        if col == 0:
            ax.text(
                -0.08, 0.5, "Prediction",
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=12, fontweight="bold",
            )

        # 第三行：标签图
        ax = axes[2, col]
        plot_binary(
            ax,
            label_binary,
            boundary,
            extent,
            show_xticks=True,
            show_yticks=(col == 0),
        )
        if col == 0:
            ax.text(
                -0.08, 0.5, "Label",
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=12, fontweight="bold",
            )

    figure.suptitle(
        f"Historical prediction grid - {model_name}",
        fontsize=16,
    )

    if probability_image_for_cbar is not None:
        colorbar = figure.colorbar(
            probability_image_for_cbar,
            ax=axes[0, :].ravel().tolist(),
            fraction=0.018,
            pad=0.01,
        )
        colorbar.set_label("Fire probability")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{model_name}_historical_3x7.png"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print("Saved:", png_path)

    if save_pdf:
        pdf_path = output_dir / f"{model_name}_historical_3x7.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        print("Saved:", pdf_path)

    plt.close(figure)


def visualize_per_model_forecast(
    model_name: str,
    dates: Sequence[str],
    root: Path,
    expected_shape: Tuple[int, int],
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    mask_zero_probability: bool,
) -> None:
    figure, axes = plt.subplots(
        nrows=2,
        ncols=7,
        figsize=(28, 8),
        constrained_layout=True,
    )

    probability_image_for_cbar = None

    vmax = []
    vmin = []
    for _, date_text in enumerate(dates):
        probability_path, binary_path = get_prediction_paths(
            root,
            model_name,
            date_text,
        )

        probability = load_probability_map(probability_path, expected_shape)
        vmax.append(probability.max())
        vmin.append(probability.min())
    vmax = max(vmax)
    vmin = min(vmin)

    for col, date_text in enumerate(dates):
        probability_path, binary_path = get_prediction_paths(
            root,
            model_name,
            date_text,
        )

        probability = load_probability_map(probability_path, expected_shape)
        predicted_binary = load_binary_map(binary_path, expected_shape)

        ax = axes[0, col]
        probability_image_for_cbar = plot_probability(
            ax,
            probability,
            boundary,
            extent,
            show_xticks=False,
            show_yticks=(col == 0),
            mask_zero_probability=mask_zero_probability,
            vmin=vmin, 
            vmax=vmax,
        )
        ax.set_title(date_text, fontsize=11)
        if col == 0:
            ax.text(
                -0.08, 0.5, "Probability",
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=12, fontweight="bold",
            )

        ax = axes[1, col]
        plot_binary(
            ax,
            predicted_binary,
            boundary,
            extent,
            show_xticks=True,
            show_yticks=(col == 0),
        )
        if col == 0:
            ax.text(
                -0.08, 0.5, "Prediction",
                transform=ax.transAxes,
                rotation=90,
                va="center", ha="center",
                fontsize=12, fontweight="bold",
            )

    figure.suptitle(
        f"Forecast prediction grid - {model_name}",
        fontsize=16,
    )

    if probability_image_for_cbar is not None:
        colorbar = figure.colorbar(
            probability_image_for_cbar,
            ax=axes[0, :].ravel().tolist(),
            fraction=0.018,
            pad=0.01,
        )
        colorbar.set_label("Fire probability")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{model_name}_forecast_2x7.png"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print("Saved:", png_path)

    if save_pdf:
        pdf_path = output_dir / f"{model_name}_forecast_2x7.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        print("Saved:", pdf_path)

    plt.close(figure)


# =============================================================================
# 8. 模型比较图（历史）
# =============================================================================
def visualize_compare_probability(
    model_names: Sequence[str],
    dates: Sequence[str],
    historical_root: Path,
    label_dir: Path,
    label_pattern: str,
    label_threshold: float,
    expected_shape: Tuple[int, int],
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
    mask_zero_probability: bool,
) -> None:
    """
    绘制历史时期不同模型的概率图比较。

    每个模型使用独立的颜色范围和独立的 colorbar：
        - vmin：该模型在全部历史日期中的最小概率；
        - vmax：该模型在全部历史日期中的最大概率。

    这样可以避免不同模型概率尺度差异较大时，使用统一颜色范围
    导致部分模型的空间差异难以辨认。
    """
    ncols = len(model_names) + 1
    figure, axes = plt.subplots(
        nrows=len(dates),
        ncols=ncols,
        figsize=(4.8 * ncols, 3.8 * len(dates)),
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------
    # 1. 一次性读取各模型全部日期的概率图，并分别计算颜色范围。
    #    同时避免后续绘图时重复读取同一文件。
    # -------------------------------------------------------------------------
    probability_maps = {}
    probability_ranges = {}

    for model_name in model_names:
        model_maps = []
        model_vmin = np.inf
        model_vmax = -np.inf

        for date_text in dates:
            probability_path, _ = get_prediction_paths(
                historical_root,
                model_name,
                date_text,
            )
            probability = load_probability_map(
                probability_path,
                expected_shape,
            )

            model_maps.append(probability)
            model_vmin = min(
                model_vmin,
                float(np.min(probability)),
            )
            model_vmax = max(
                model_vmax,
                float(np.max(probability)),
            )

        if not (
            np.isfinite(model_vmin)
            and np.isfinite(model_vmax)
        ):
            raise ValueError(
                f"{model_name} 的概率范围包含非有限值："
                f"vmin={model_vmin}, vmax={model_vmax}"
            )

        # 若全部日期、全部像元概率完全相同，给颜色范围增加极小宽度，
        # 防止 Matplotlib 出现 vmin == vmax 的归一化问题。
        if np.isclose(model_vmin, model_vmax):
            if model_vmin <= 0.0:
                display_vmin = 0.0
                display_vmax = 1e-6
            elif model_vmax >= 1.0:
                display_vmin = 1.0 - 1e-6
                display_vmax = 1.0
            else:
                epsilon = max(
                    1e-6,
                    abs(model_vmin) * 1e-6,
                )
                display_vmin = max(
                    0.0,
                    model_vmin - epsilon,
                )
                display_vmax = min(
                    1.0,
                    model_vmax + epsilon,
                )
        else:
            display_vmin = model_vmin
            display_vmax = model_vmax

        probability_maps[model_name] = model_maps
        probability_ranges[model_name] = (
            display_vmin,
            display_vmax,
            model_vmin,
            model_vmax,
        )

        print(
            f"[模型比较概率范围] "
            f"{COMPARE_MODEL_DISPLAY_NAMES.get(model_name, model_name)}："
            f"最小值={model_vmin:.8f}，最大值={model_vmax:.8f}"
        )

    # 保存每个模型对应的图像对象，后面分别创建 colorbar。
    model_mappables = {}

    # -------------------------------------------------------------------------
    # 2. 绘制各日期、各模型概率图。
    # -------------------------------------------------------------------------
    for row, date_text in enumerate(dates):
        for col, model_name in enumerate(model_names):
            probability = probability_maps[model_name][row]
            (
                display_vmin,
                display_vmax,
                _,
                _,
            ) = probability_ranges[model_name]

            ax = axes[row, col]
            image = plot_probability(
                ax,
                probability,
                boundary,
                extent,
                show_xticks=(row == len(dates) - 1),
                show_yticks=(col == 0),
                mask_zero_probability=mask_zero_probability,
                vmin=display_vmin,
                vmax=display_vmax,
            )

            # 同一模型的所有日期共用同一归一化范围，
            # 任意一天返回的图像对象均可用于生成该模型的 colorbar。
            model_mappables[model_name] = image

            # 放大经纬度刻度字号。
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=COMPARE_TICK_FONTSIZE,
            )

            if row == 0:
                ax.set_title(
                    COMPARE_MODEL_DISPLAY_NAMES.get(
                        model_name,
                        model_name,
                    ),
                    fontsize=COMPARE_TITLE_FONTSIZE,
                    # fontweight="bold",
                    pad=10,
                )

            if col == 0:
                ax.text(
                    -0.33,
                    0.5,
                    format_date_chinese(date_text),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=COMPARE_DATE_FONTSIZE,
                    # fontweight="bold",
                )

        # 最后一列绘制真实标签。
        label_path = get_label_path(
            label_dir,
            label_pattern,
            date_text,
        )
        label_binary = load_label_binary(
            label_path,
            expected_shape,
            label_threshold,
        )

        ax = axes[row, ncols - 1]
        plot_binary(
            ax,
            label_binary,
            boundary,
            extent,
            show_xticks=(row == len(dates) - 1),
            show_yticks=False,
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=COMPARE_TICK_FONTSIZE,
        )

        if row == 0:
            ax.set_title(
                "GT",
                fontsize=COMPARE_TITLE_FONTSIZE,
                # fontweight="bold",
                pad=10,
            )

    # 按要求不设置总标题（suptitle）。

    # -------------------------------------------------------------------------
    # 3. 每个模型单独创建一个水平 colorbar。
    #    colorbar 的范围对应该模型全部日期的最小值和最大值。
    # -------------------------------------------------------------------------
    for col, model_name in enumerate(model_names):
        image = model_mappables.get(model_name)

        if image is None:
            continue
        
        # 非校准
        colorbar = figure.colorbar(image,ax=axes[:, col].ravel().tolist(),orientation="horizontal",fraction=0.025,pad=0.025,aspect=28,)
        colorbar.set_label("Fire Probability",fontsize=COMPARE_COLORBAR_LABEL_FONTSIZE,)# fontweight="bold",
        colorbar.ax.tick_params(labelsize=COMPARE_COLORBAR_TICK_FONTSIZE,)

        # # 校准
        # from matplotlib.ticker import FuncFormatter, MaxNLocator
        # colorbar = figure.colorbar(image,ax=axes[:, col].ravel().tolist(),orientation="horizontal",fraction=0.025,pad=0.025,aspect=28,)
        # # 控制颜色条刻度数量，避免过于密集
        # colorbar.locator = MaxNLocator(nbins=5)
        # # 原始概率乘以1000后显示
        # colorbar.formatter = FuncFormatter(lambda value, position: (f"{value * 1000:.2f}".rstrip("0").rstrip(".")))
        # # 必须在设置 locator 和 formatter 后调用
        # colorbar.update_ticks()
        # colorbar.set_label("火灾概率（×10$^{-3}$）",fontsize=COMPARE_COLORBAR_LABEL_FONTSIZE,)
        # colorbar.ax.tick_params(axis="x",labelsize=COMPARE_COLORBAR_TICK_FONTSIZE,)
        # # 隐藏 Matplotlib 自动生成的科学计数法偏移文字
        # colorbar.ax.xaxis.get_offset_text().set_visible(False)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    png_path = (
        output_dir
        / "historical_compare_probability_7rows.png"
    )
    figure.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    print("Saved:", png_path)

    # if save_pdf:
    pdf_path = (
        output_dir
        / "historical_compare_probability_7rows.pdf"
    )
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )
    print("Saved:", pdf_path)

    plt.close(figure)

def visualize_compare_binary(
    model_names: Sequence[str],
    dates: Sequence[str],
    historical_root: Path,
    label_dir: Path,
    label_pattern: str,
    label_threshold: float,
    expected_shape: Tuple[int, int],
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    ncols = len(model_names) + 1
    figure, axes = plt.subplots(
        nrows=7,
        ncols=ncols,
        figsize=(4.8 * ncols, 3.3 * 7),
        constrained_layout=True,
    )

    for row, date_text in enumerate(dates):
        for col, model_name in enumerate(model_names):
            _, binary_path = get_prediction_paths(
                historical_root,
                model_name,
                date_text,
            )

            predicted_binary = load_binary_map(binary_path, expected_shape)

            ax = axes[row, col]
            plot_binary(
                ax,
                predicted_binary,
                boundary,
                extent,
                show_xticks=(row == len(dates) - 1),
                show_yticks=(col == 0),
            )

            # 放大经纬度刻度字号。
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=COMPARE_TICK_FONTSIZE,
            )

            if row == 0:
                ax.set_title(
                    COMPARE_MODEL_DISPLAY_NAMES.get(
                        model_name,
                        model_name,
                    ),
                    fontsize=COMPARE_TITLE_FONTSIZE,
                    # fontweight="bold",
                    pad=10,
                )

            if col == 0:
                ax.text(
                    -0.33,
                    0.5,
                    format_date_chinese(date_text),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=COMPARE_DATE_FONTSIZE,
                    # fontweight="bold",
                )

        label_path = get_label_path(
            label_dir,
            label_pattern,
            date_text,
        )
        label_binary = load_label_binary(
            label_path,
            expected_shape,
            label_threshold,
        )

        ax = axes[row, ncols - 1]
        plot_binary(
            ax,
            label_binary,
            boundary,
            extent,
            show_xticks=(row == len(dates) - 1),
            show_yticks=False,
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=COMPARE_TICK_FONTSIZE,
        )

        if row == 0:
            ax.set_title(
                "真实标签",
                fontsize=COMPARE_TITLE_FONTSIZE,
                # fontweight="bold",
                pad=10,
            )

    # 按要求不设置总标题（suptitle）。

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "historical_compare_binary_7rows.png"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print("Saved:", png_path)

    if save_pdf:
        pdf_path = output_dir / "historical_compare_binary_7rows.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        print("Saved:", pdf_path)

    plt.close(figure)


# =============================================================================
# 8b. 月尺度模型比较图
# =============================================================================
def load_guangdong_mask(path: Path, expected_shape: Tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到广东掩膜：{path}")

    mask = np.load(path).astype(bool)
    if mask.shape != expected_shape:
        raise ValueError(
            f"广东掩膜 shape={mask.shape}，期望 {expected_shape}"
        )
    return mask


def monthly_probability_statistics(
    root: Path,
    model_name: str,
    dates: Sequence[str],
    expected_shape: Tuple[int, int],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回月平均概率图和全省逐日平均概率。"""
    total = np.zeros(expected_shape, dtype=np.float64)
    daily_means: List[float] = []

    for date_text in dates:
        probability_path, _ = get_prediction_paths(
            root,
            model_name,
            date_text,
        )
        probability = load_probability_map(
            probability_path,
            expected_shape,
        )
        total += probability
        daily_means.append(float(probability[mask].mean()))

    monthly_mean = (total / len(dates)).astype(np.float32)
    return monthly_mean, np.asarray(daily_means, dtype=np.float64)


def monthly_binary_day_count(
    root: Path,
    model_name: str,
    dates: Sequence[str],
    expected_shape: Tuple[int, int],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回月内判定为火灾的天数图和全省逐日火灾像元数。"""
    counts = np.zeros(expected_shape, dtype=np.uint8)
    daily_counts: List[int] = []

    for date_text in dates:
        _, binary_path = get_prediction_paths(
            root,
            model_name,
            date_text,
        )
        binary = load_binary_map(binary_path, expected_shape)
        counts += binary
        daily_counts.append(int(binary[mask].sum()))

    return counts, np.asarray(daily_counts, dtype=np.int64)


def monthly_label_day_count(
    label_dir: Path,
    label_pattern: str,
    dates: Sequence[str],
    expected_shape: Tuple[int, int],
    mask: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回真实火点月内出现天数和逐日真实火点像元数。"""
    counts = np.zeros(expected_shape, dtype=np.uint8)
    daily_counts: List[int] = []

    for date_text in dates:
        label_path = get_label_path(
            label_dir,
            label_pattern,
            date_text,
        )
        label_binary = load_label_binary(
            label_path,
            expected_shape,
            threshold,
        )
        counts += label_binary
        daily_counts.append(int(label_binary[mask].sum()))

    return counts, np.asarray(daily_counts, dtype=np.int64)


def plot_count_map(
    axis,
    counts: np.ndarray,
    mask: np.ndarray,
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    show_xticks: bool,
    show_yticks: bool,
    vmax: int,
):
    display = np.ma.masked_where(~mask, counts)
    image = axis.imshow(
        display,
        extent=extent,
        origin="upper",
        vmin=0,
        vmax=max(1, int(vmax)),
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )
    draw_map_common(
        axis,
        boundary,
        extent,
        show_xticks=show_xticks,
        show_yticks=show_yticks,
    )
    return image


def _monthly_axis_style(axis) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=COMPARE_TICK_FONTSIZE,
    )


def set_compact_probability_colorbar(colorbar, vmax: float) -> None:
    """压缩概率色标刻度，避免小概率时刻度文字相互重叠。"""
    if np.isfinite(vmax) and 0 < vmax < 0.01:
        exponent = int(math.floor(math.log10(vmax) / 3.0)) * 3
        scale = 10.0 ** (-exponent)
        colorbar.locator = MaxNLocator(nbins=4)
        colorbar.formatter = FuncFormatter(
            lambda value, position: f"{value * scale:.0f}"
        )
        colorbar.update_ticks()
        colorbar.ax.xaxis.get_offset_text().set_visible(False)
        colorbar.set_label(
            f"Fire Probability (×10$^{{{exponent}}}$)",
            fontsize=13,
            labelpad=2,
        )
    else:
        colorbar.locator = MaxNLocator(nbins=4)
        colorbar.formatter = FuncFormatter(
            lambda value, position: (
                f"{value:.2f}".rstrip("0").rstrip(".")
            )
        )
        colorbar.update_ticks()
        colorbar.set_label(
            "Fire Probability",
            fontsize=13,
            labelpad=2,
        )

    colorbar.ax.tick_params(labelsize=11)



# =============================================================================
def visualize_monthly_probability_comparison(
    model_names: Sequence[str],
    dates: Sequence[str],
    historical_root: Path,
    expected_shape: Tuple[int, int],
    mask: np.ndarray,
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    output_dir: Path,
    dpi: int,
    mask_zero_probability: bool,
) -> None:
    """上排：各模型月平均概率；下排：全省逐日平均概率曲线。"""
    ncols = len(model_names)
    probability_maps: Dict[str, np.ndarray] = {}
    daily_series: Dict[str, np.ndarray] = {}
    ranges: Dict[str, Tuple[float, float]] = {}

    for model_name in model_names:
        monthly_mean, daily_means = monthly_probability_statistics(
            historical_root,
            model_name,
            dates,
            expected_shape,
            mask,
        )
        probability_maps[model_name] = monthly_mean
        daily_series[model_name] = daily_means

        values = monthly_mean[mask]
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if np.isclose(vmin, vmax):
            epsilon = max(1e-8, abs(vmin) * 1e-6)
            vmin = max(0.0, vmin - epsilon)
            vmax = min(1.0, vmax + epsilon)
        ranges[model_name] = (vmin, vmax)

        print(
            f"[月平均概率范围] "
            f"{COMPARE_MODEL_DISPLAY_NAMES.get(model_name, model_name)}："
            f"最小值={vmin:.8f}，最大值={vmax:.8f}"
        )

    figure = plt.figure(
        figsize=(5.4 * ncols, 7.2),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        2,
        ncols,
        height_ratios=[2.8, 1.0],
    )

    for col, model_name in enumerate(model_names):
        axis = figure.add_subplot(grid[0, col])
        vmin, vmax = ranges[model_name]
        image = plot_probability(
            axis,
            probability_maps[model_name],
            boundary,
            extent,
            show_xticks=True,
            show_yticks=(col == 0),
            mask_zero_probability=mask_zero_probability,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(
            COMPARE_MODEL_DISPLAY_NAMES.get(model_name, model_name),
            fontsize=COMPARE_TITLE_FONTSIZE,
            pad=10,
        )
        _monthly_axis_style(axis)

        colorbar = figure.colorbar(
            image,
            ax=axis,
            orientation="horizontal",
            fraction=0.055,
            pad=0.02,
        )
        set_compact_probability_colorbar(colorbar, vmax)

    time_axis = figure.add_subplot(grid[1, :])
    day_numbers = np.arange(1, len(dates) + 1)
    for model_name in model_names:
        time_axis.plot(
            day_numbers,
            daily_series[model_name],
            color=MODEL_COLORS.get(model_name),
            linewidth=1.8,
            label=COMPARE_MODEL_DISPLAY_NAMES.get(
                model_name,
                model_name,
            ),
        )

    time_axis.set_xlim(0.5, len(dates) + 0.5)
    time_axis.set_xlabel("December 2025 day", fontsize=16)
    time_axis.set_ylabel("Province mean\nfire probability", fontsize=16)
    time_axis.grid(True, alpha=0.3)
    time_axis.tick_params(axis="both", labelsize=13)
    time_axis.legend(
        ncol=min(len(model_names), 4),
        fontsize=13,
        frameon=False,
        loc="upper right",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "historical_compare_probability_month.png"
    pdf_path = output_dir / "historical_compare_probability_month.pdf"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    print("Saved:", png_path)
    print("Saved:", pdf_path)
    plt.close(figure)


def visualize_monthly_binary_comparison(
    model_names: Sequence[str],
    dates: Sequence[str],
    historical_root: Path,
    label_dir: Path,
    label_pattern: str,
    label_threshold: float,
    expected_shape: Tuple[int, int],
    mask: np.ndarray,
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    output_dir: Path,
    dpi: int,
    save_pdf: bool,
) -> None:
    """上排：各模型月内火灾判定日数；下排：逐日火灾像元数曲线。"""
    count_maps: Dict[str, np.ndarray] = {}
    daily_series: Dict[str, np.ndarray] = {}

    for model_name in model_names:
        counts, daily_counts = monthly_binary_day_count(
            historical_root,
            model_name,
            dates,
            expected_shape,
            mask,
        )
        count_maps[model_name] = counts
        daily_series[model_name] = daily_counts

    label_counts, label_daily_counts = monthly_label_day_count(
        label_dir,
        label_pattern,
        dates,
        expected_shape,
        mask,
        label_threshold,
    )
    count_maps["__label__"] = label_counts
    daily_series["__label__"] = label_daily_counts

    vmax = int(
        max(
            int(np.max(counts[mask]))
            for counts in count_maps.values()
        )
    )

    panel_names = list(model_names) + ["__label__"]
    ncols = len(panel_names)

    figure = plt.figure(
        figsize=(5.0 * ncols, 7.2),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        2,
        ncols,
        height_ratios=[2.8, 1.0],
    )

    top_axes = []
    last_image = None
    for col, panel_name in enumerate(panel_names):
        axis = figure.add_subplot(grid[0, col])
        top_axes.append(axis)

        last_image = plot_count_map(
            axis,
            count_maps[panel_name],
            mask,
            boundary,
            extent,
            show_xticks=True,
            show_yticks=(col == 0),
            vmax=vmax,
        )

        title = (
            "GT"
            if panel_name == "__label__"
            else COMPARE_MODEL_DISPLAY_NAMES.get(
                panel_name,
                panel_name,
            )
        )
        axis.set_title(
            title,
            fontsize=COMPARE_TITLE_FONTSIZE,
            pad=10,
        )
        _monthly_axis_style(axis)

    colorbar = figure.colorbar(
        last_image,
        ax=top_axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
    )
    colorbar.set_label(
        "Fire days in month",
        fontsize=COMPARE_COLORBAR_LABEL_FONTSIZE,
    )
    colorbar.ax.tick_params(
        labelsize=COMPARE_COLORBAR_TICK_FONTSIZE,
    )

    time_axis = figure.add_subplot(grid[1, :])
    day_numbers = np.arange(1, len(dates) + 1)

    for model_name in model_names:
        time_axis.plot(
            day_numbers,
            daily_series[model_name],
            color=MODEL_COLORS.get(model_name),
            linewidth=1.8,
            label=COMPARE_MODEL_DISPLAY_NAMES.get(
                model_name,
                model_name,
            ),
        )

    time_axis.plot(
        day_numbers,
        daily_series["__label__"],
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="GT",
    )

    time_axis.set_xlim(0.5, len(dates) + 0.5)
    time_axis.set_xlabel("December 2025 day", fontsize=16)
    time_axis.set_ylabel("Fire pixels\nper day", fontsize=16)
    time_axis.grid(True, alpha=0.3)
    time_axis.tick_params(axis="both", labelsize=13)
    time_axis.legend(
        ncol=min(len(model_names) + 1, 5),
        fontsize=12,
        frameon=False,
        loc="upper right",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "historical_compare_binary_month.png"
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print("Saved:", png_path)

    if save_pdf:
        pdf_path = output_dir / "historical_compare_binary_month.pdf"
        figure.savefig(pdf_path, bbox_inches="tight")
        print("Saved:", pdf_path)

    plt.close(figure)


# 9. 主流程
# =============================================================================
def main():
    args = parse_args()

    historical_root = Path(args.historical_root)
    forecast_root = Path(args.forecast_root)
    label_dir = Path(args.label_dir)
    output_root = Path(args.output_root)

    hist_dates = date_range(args.hist_start_date, args.hist_end_date)
    forecast_dates = date_range(args.forecast_start_date, args.forecast_end_date)

    require_dates(hist_dates, "历史日期范围")
    require_dates(forecast_dates, "预测日期范围")

    extent, expected_shape = load_extent_and_shape(
        Path(args.grid_metadata)
    )
    boundary = load_boundary(
        Path(args.boundary_shp)
    )
    mask = load_guangdong_mask(
        Path(args.mask),
        expected_shape,
    )

    # -------------------------------------------------------------------------
    # A. 每个模型：历史图 + 预测图（本报告默认关闭）
    # -------------------------------------------------------------------------
    if GENERATE_PER_MODEL_FIGURES:
        per_model_hist_dir = output_root / "per_model_historical"
        per_model_forecast_dir = output_root / "per_model_forecast"

        for model_name in PER_MODEL_NAMES:
            print("\n[PER-MODEL HISTORICAL]", model_name)
            visualize_per_model_historical(
                model_name=model_name,
                dates=hist_dates,
                root=historical_root,
                label_dir=label_dir,
                label_pattern=args.label_pattern,
                label_threshold=args.label_threshold,
                expected_shape=expected_shape,
                boundary=boundary,
                extent=extent,
                output_dir=per_model_hist_dir,
                dpi=args.dpi,
                save_pdf=args.save_pdf,
                mask_zero_probability=args.mask_zero_probability,
            )

            print("[PER-MODEL FORECAST]", model_name)
            visualize_per_model_forecast(
                model_name=model_name,
                dates=forecast_dates,
                root=forecast_root,
                expected_shape=expected_shape,
                boundary=boundary,
                extent=extent,
                output_dir=per_model_forecast_dir,
                dpi=args.dpi,
                save_pdf=args.save_pdf,
                mask_zero_probability=args.mask_zero_probability,
            )

    # -------------------------------------------------------------------------
    # B. 历史模型比较图：月平均概率图和月内火灾判定日数图各 1 张
    # -------------------------------------------------------------------------
    compare_dir = output_root / "historical_comparison"

    print("\n[COMPARE] monthly probability")
    visualize_monthly_probability_comparison(
        model_names=COMPARE_MODEL_NAMES,
        dates=hist_dates,
        historical_root=historical_root,
        expected_shape=expected_shape,
        mask=mask,
        boundary=boundary,
        extent=extent,
        output_dir=compare_dir,
        dpi=args.dpi,
        mask_zero_probability=args.mask_zero_probability,
    )

    print("[COMPARE] monthly binary day count")
    visualize_monthly_binary_comparison(
        model_names=COMPARE_MODEL_NAMES,
        dates=hist_dates,
        historical_root=historical_root,
        label_dir=label_dir,
        label_pattern=args.label_pattern,
        label_threshold=args.label_threshold,
        expected_shape=expected_shape,
        mask=mask,
        boundary=boundary,
        extent=extent,
        output_dir=compare_dir,
        dpi=args.dpi,
        save_pdf=args.save_pdf,
    )

    print("\n[DONE]")
    print("Output root:", output_root.resolve())
    print("Generated comparison figures: 2")


if __name__ == "__main__":
    main()