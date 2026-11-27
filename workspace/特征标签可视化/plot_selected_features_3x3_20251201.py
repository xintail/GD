#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制广东省 2025 年 12 月九项典型驱动因子的月平均 3×3 空间分布图。

九项变量：
1. 平均海平面气压
2. 10 米东西向风
3. 10 米南北向风
4. 2 米气温
5. 1000 百帕位势
6. 1000 百帕比湿
7. 1000 百帕气温
8. 叶面积指数
9. 人口密度

说明：
- 默认对 2025-12-01 至 2025-12-31 逐日特征文件求月平均。
- 特征文件形状应为 (72, H, W)。
- 通道信息从 feature_channel_metadata.json 的 channel_order 字段读取。
- 仅最左列显示纬度，最下行显示经度。
- 每个子图使用独立颜色条，以适应不同变量的数值范围。
- 可选择在每个子图左上角显示（a）～（i）的编号。
"""

from __future__ import annotations

import argparse
import json
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# =============================================================================
# 1. 默认配置
# =============================================================================
# 默认统计时段：2025 年 12 月整月。
START_DATE = "20251201"
END_DATE = "20251231"

# 输出文件中的日期标签。月平均结果使用“202512月均”。
OUTPUT_DATE_LABEL = "202512月均"

FEATURE_DIR = Path("./data/processed/features_1km")
FEATURE_PATTERN = "feature_{date}.npy"

CHANNEL_METADATA_PATH = Path(
    "./data/processed/feature_channel_metadata.json"
)
GRID_METADATA_PATH = Path(
    "./data/processed/target_grid_metadata.json"
)
MASK_PATH = Path(
    "./data/processed/guangdong_mask_1km.npy"
)
BOUNDARY_SHP = Path(
    "./data/boundary/gd_boundary.shp"
)

OUTPUT_DIR = Path(
    "./workspace/特征标签可视化/selected_features_3x3"
)
# OUTPUT_PATTERN = "E广东省典型驱动因子_{date}_3乘3.pdf"
OUTPUT_PATTERN = "E广东省典型驱动因子_{date}_3乘3.png"

DPI = 300
FIGSIZE = (18, 12)

# 字号
MAIN_TITLE_SIZE = 23
SUBTITLE_SIZE = 18
TICK_SIZE = 14
COLORBAR_TICK_SIZE = 13
COLORBAR_LABEL_SIZE = 14

# 子图编号设置。True 表示在每个子图左上角显示（a）～（i）。
SHOW_PANEL_LABELS = True
PANEL_LABEL_SIZE = 19
PANEL_LABEL_X = 0.025
PANEL_LABEL_Y = 0.975

# 中文字体。Windows 下通常优先使用微软雅黑或黑体。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 2. 待绘制变量
# =============================================================================
# source、variable 和 pressure_level_hpa 与实际元数据结构保持一致。
# 中文
SELECTED_VARIABLES: List[Dict[str, object]] = [
    {
        "title": "平均海平面气压",
        "unit": "帕",
        "source": "weather_surface",
        "variable": "msl",
        "pressure_level_hpa": None,
    },
    {
        "title": "10米东西向风",
        "unit": "米/秒",
        "source": "weather_surface",
        "variable": "10u",
        "pressure_level_hpa": None,
    },
    {
        "title": "10米南北向风",
        "unit": "米/秒",
        "source": "weather_surface",
        "variable": "10v",
        "pressure_level_hpa": None,
    },
    {
        "title": "2米气温",
        "unit": "开尔文",
        "source": "weather_surface",
        "variable": "2t",
        "pressure_level_hpa": None,
    },
    {
        "title": "1000百帕位势",
        "unit": "平方米/平方秒",
        "source": "weather_upper",
        "variable": "z",
        "pressure_level_hpa": 1000,
    },
    {
        "title": "1000百帕比湿",
        "unit": "千克/千克",
        "source": "weather_upper",
        "variable": "q",
        "pressure_level_hpa": 1000,
    },
    {
        "title": "1000百帕气温",
        "unit": "开尔文",
        "source": "weather_upper",
        "variable": "t",
        "pressure_level_hpa": 1000,
    },
    {
        "title": "叶面积指数",
        "unit": "",
        "source": "vegetation",
        "variable": "Lai",
        "pressure_level_hpa": None,
    },
    {
        "title": "人口密度",
        "unit": "",
        "source": "population",
        "variable": "population",
        "pressure_level_hpa": None,
    },
]


# SELECTED_VARIABLES: List[Dict[str, object]] = [
#     {
#         "title": "Mean sea level pressure",
#         "unit": "Pa",
#         "source": "weather_surface",
#         "variable": "msl",
#         "pressure_level_hpa": None,
#     },
#     {
#         "title": "10m u-component of wind",
#         "unit": "m/s",
#         "source": "weather_surface",
#         "variable": "10u",
#         "pressure_level_hpa": None,
#     },
#     {
#         "title": "10m v-component of wind",
#         "unit": "m/s",
#         "source": "weather_surface",
#         "variable": "10v",
#         "pressure_level_hpa": None,
#     },
#     {
#         "title": "2m temperature",
#         "unit": "K",
#         "source": "weather_surface",
#         "variable": "2t",
#         "pressure_level_hpa": None,
#     },
#     {
#         "title": "1000 hPa geopotential",
#         "unit": "$m^2/s^2$",
#         "source": "weather_upper",
#         "variable": "z",
#         "pressure_level_hpa": 1000,
#     },
#     {
#         "title": "1000 hPa specific humidity",
#         "unit": "kg/kg",
#         "source": "weather_upper",
#         "variable": "q",
#         "pressure_level_hpa": 1000,
#     },
#     {
#         "title": "1000 hPa temperature",
#         "unit": "K",
#         "source": "weather_upper",
#         "variable": "t",
#         "pressure_level_hpa": 1000,
#     },
#     {
#         "title": "Leaf area index",
#         "unit": "",
#         "source": "vegetation",
#         "variable": "Lai",
#         "pressure_level_hpa": None,
#     },
#     {
#         "title": "Population density",
#         "unit": "",
#         "source": "population",
#         "variable": "population",
#         "pressure_level_hpa": None,
#     },
# ]

# =============================================================================
# 3. 参数与数据读取
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="绘制广东省九项典型驱动因子的三乘三空间分布图。"
    )
    parser.add_argument(
        "--start-date",
        default=START_DATE,
        help="统计起始日期，格式 YYYYMMDD。",
    )
    parser.add_argument(
        "--end-date",
        default=END_DATE,
        help="统计结束日期，格式 YYYYMMDD（包含当天）。",
    )
    parser.add_argument(
        "--date",
        default=OUTPUT_DATE_LABEL,
        help="输出文件名中使用的日期标签。",
    )
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--feature-pattern", default=FEATURE_PATTERN)
    parser.add_argument(
        "--channel-metadata",
        default=str(CHANNEL_METADATA_PATH),
    )
    parser.add_argument(
        "--grid-metadata",
        default=str(GRID_METADATA_PATH),
    )
    parser.add_argument("--mask", default=str(MASK_PATH))
    parser.add_argument("--boundary-shp", default=str(BOUNDARY_SHP))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=DPI)

    # 子图编号开关：
    # 默认按照 SHOW_PANEL_LABELS 配置显示；
    # 命令行可用 --panel-labels 强制显示，
    # 或用 --no-panel-labels 关闭。
    panel_group = parser.add_mutually_exclusive_group()
    panel_group.add_argument(
        "--panel-labels",
        dest="show_panel_labels",
        action="store_true",
        help="在每个子图左上角显示（a）至（i）的编号。",
    )
    panel_group.add_argument(
        "--no-panel-labels",
        dest="show_panel_labels",
        action="store_false",
        help="不显示子图编号。",
    )
    parser.set_defaults(
        show_panel_labels=SHOW_PANEL_LABELS
    )

    return parser.parse_args()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_features(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到特征文件：{path}")

    features = np.load(path)
    if features.ndim != 3:
        raise ValueError(
            f"特征数组应为三维 (通道, 高度, 宽度)，当前为 {features.shape}"
        )

    return np.asarray(features)


def date_range(start_date: str, end_date: str) -> List[str]:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    if end < start:
        raise ValueError("结束日期不能早于开始日期。")

    dates: List[str] = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    return dates


def load_monthly_features(
    feature_dir: Path,
    feature_pattern: str,
    start_date: str,
    end_date: str,
    selected_channels: List[int],
    expected_shape: Tuple[int, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    逐日读取特征文件，仅累加选中的通道，最后求月平均。

    返回：
        monthly_features: shape=(len(selected_channels), H, W)
        dates: 参与统计的日期列表
    """
    dates = date_range(start_date, end_date)
    monthly_sum: Optional[np.ndarray] = None

    for date_text in dates:
        path = feature_dir / feature_pattern.format(date=date_text)
        if not path.exists():
            raise FileNotFoundError(f"找不到特征文件：{path}")

        features = np.load(path, mmap_mode="r")
        if features.ndim != 3:
            raise ValueError(
                f"特征数组应为三维 (通道, 高度, 宽度)，"
                f"当前为 {features.shape}：{path}"
            )

        if tuple(features.shape[1:]) != tuple(expected_shape):
            raise ValueError(
                f"{path} 空间形状 {features.shape[1:]} "
                f"与掩膜形状 {expected_shape} 不一致。"
            )

        if monthly_sum is None:
            monthly_sum = np.zeros(
                (len(selected_channels),) + tuple(expected_shape),
                dtype=np.float64,
            )

        for index, channel in enumerate(selected_channels):
            if not 0 <= channel < features.shape[0]:
                raise IndexError(
                    f"通道 {channel} 超出 {path} 的范围 "
                    f"0～{features.shape[0] - 1}。"
                )

            data = np.asarray(features[channel], dtype=np.float64)
            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            monthly_sum[index] += data

        del features

    if monthly_sum is None:
        raise ValueError("日期范围内没有可用的特征文件。")

    monthly_features = (monthly_sum / len(dates)).astype(np.float32)
    return monthly_features, dates


def build_extent(
    grid_metadata: Dict,
) -> Tuple[float, float, float, float]:
    """读取实际 target_grid_metadata.json 的 bounds 字典。"""
    if "bounds" not in grid_metadata:
        raise KeyError("网格元数据缺少 bounds 字段。")

    bounds = grid_metadata["bounds"]

    # 实际脚本所生成的形式：
    # {"left": ..., "right": ..., "bottom": ..., "top": ...}
    if isinstance(bounds, dict):
        required = ["left", "right", "bottom", "top"]
        missing = [key for key in required if key not in bounds]
        if missing:
            raise KeyError(
                f"bounds 字段缺少以下键：{missing}"
            )
        return (
            float(bounds["left"]),
            float(bounds["right"]),
            float(bounds["bottom"]),
            float(bounds["top"]),
        )

    # 兼容 [left, bottom, right, top] 形式。
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        left, bottom, right, top = bounds
        return float(left), float(right), float(bottom), float(top)

    raise ValueError(
        "无法识别 target_grid_metadata.json 中 bounds 的结构。"
    )


# =============================================================================
# 4. 通道定位
# =============================================================================
def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def get_channel_order(channel_metadata: Dict) -> List[Dict[str, object]]:
    """
    读取实际元数据结构：
        channel_metadata["channel_order"] = [
            {
                "channel": 0,
                "source": "weather_surface",
                "variable": "msl",
                "pressure_level_hpa": None
            },
            ...
        ]
    """
    if "channel_order" not in channel_metadata:
        raise KeyError(
            "通道元数据缺少 channel_order 字段。"
            "请确认使用的是生成特征时保存的 feature_channel_metadata.json。"
        )

    order = channel_metadata["channel_order"]
    if not isinstance(order, list) or not order:
        raise ValueError("channel_order 应为非空列表。")

    for position, item in enumerate(order):
        if not isinstance(item, dict):
            raise TypeError(
                f"channel_order 第 {position} 项不是字典：{item!r}"
            )
        for key in ["channel", "source", "variable"]:
            if key not in item:
                raise KeyError(
                    f"channel_order 第 {position} 项缺少 {key} 字段。"
                )

    return order


def find_channel_index(
    channel_order: List[Dict[str, object]],
    source: str,
    variable: str,
    pressure_level_hpa: Optional[int],
) -> int:
    """按数据源、变量名和气压层精确定位通道。"""
    matches: List[int] = []

    for item in channel_order:
        item_source = normalize_text(item.get("source"))
        item_variable = normalize_text(item.get("variable"))
        item_level = item.get("pressure_level_hpa", None)

        source_ok = item_source == normalize_text(source)
        variable_ok = item_variable == normalize_text(variable)

        if pressure_level_hpa is None:
            level_ok = item_level is None
        else:
            try:
                level_ok = int(item_level) == int(pressure_level_hpa)
            except (TypeError, ValueError):
                level_ok = False

        if source_ok and variable_ok and level_ok:
            matches.append(int(item["channel"]))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise KeyError(
            "无法在 channel_order 中找到通道："
            f"source={source!r}, variable={variable!r}, "
            f"pressure_level_hpa={pressure_level_hpa!r}"
        )

    raise RuntimeError(
        "同一变量匹配到多个通道："
        f"source={source!r}, variable={variable!r}, "
        f"pressure_level_hpa={pressure_level_hpa!r}, "
        f"channels={matches}"
    )


def resolve_selected_channels(
    channel_metadata: Dict,
) -> List[int]:
    order = get_channel_order(channel_metadata)
    indices: List[int] = []

    for specification in SELECTED_VARIABLES:
        index = find_channel_index(
            channel_order=order,
            source=str(specification["source"]),
            variable=str(specification["variable"]),
            pressure_level_hpa=specification["pressure_level_hpa"],
        )
        indices.append(index)

    return indices


# =============================================================================
# 5. 坐标与绘图辅助
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


def choose_ticks(
    minimum: float,
    maximum: float,
    count: int = 4,
) -> np.ndarray:
    return np.linspace(minimum, maximum, count)


def set_outer_ticks(
    axis,
    extent: Tuple[float, float, float, float],
    show_longitude: bool,
    show_latitude: bool,
) -> None:
    left, right, bottom, top = extent

    if show_longitude:
        axis.set_xticks(choose_ticks(left, right, count=4))
        axis.xaxis.set_major_formatter(
            FuncFormatter(format_longitude)
        )
        axis.tick_params(
            axis="x",
            labelsize=TICK_SIZE,
            length=4,
        )
    else:
        axis.set_xticks([])

    if show_latitude:
        axis.set_yticks(choose_ticks(bottom, top, count=4))
        axis.yaxis.set_major_formatter(
            FuncFormatter(format_latitude)
        )
        axis.tick_params(
            axis="y",
            labelsize=TICK_SIZE,
            length=4,
        )
    else:
        axis.set_yticks([])

    axis.set_xlabel("")
    axis.set_ylabel("")


def clean_and_mask(
    data: np.ndarray,
    mask: np.ndarray,
) -> np.ma.MaskedArray:
    if data.shape != mask.shape:
        raise ValueError(
            f"变量形状 {data.shape} 与广东掩膜形状 {mask.shape} 不一致。"
        )

    clean = np.asarray(data, dtype=np.float64)
    clean = np.where(np.isfinite(clean), clean, np.nan)
    return np.ma.masked_where(~mask, clean)


# =============================================================================
# 6. 主流程
# =============================================================================
def main() -> None:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    channel_metadata_path = Path(args.channel_metadata)
    grid_metadata_path = Path(args.grid_metadata)
    mask_path = Path(args.mask)
    boundary_shp = Path(args.boundary_shp)
    output_dir = Path(args.output_dir)

    for required_path, label in [
        (channel_metadata_path, "通道元数据"),
        (grid_metadata_path, "网格元数据"),
        (mask_path, "广东掩膜"),
        (boundary_shp, "广东边界"),
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"找不到{label}：{required_path}"
            )

    channel_metadata = load_json(channel_metadata_path)
    grid_metadata = load_json(grid_metadata_path)
    extent = build_extent(grid_metadata)

    mask = np.load(mask_path).astype(bool)

    selected_channels = resolve_selected_channels(
        channel_metadata
    )

    features, feature_dates = load_monthly_features(
        feature_dir=feature_dir,
        feature_pattern=args.feature_pattern,
        start_date=args.start_date,
        end_date=args.end_date,
        selected_channels=selected_channels,
        expected_shape=mask.shape,
    )

    if features.shape[1:] != mask.shape:
        raise ValueError(
            f"特征空间形状 {features.shape[1:]} 与掩膜形状 {mask.shape} 不一致。"
        )

    print(
        "统计时段：",
        feature_dates[0],
        "至",
        feature_dates[-1],
        f"（共 {len(feature_dates)} 天）",
    )

    boundary = gpd.read_file(boundary_shp)
    if boundary.empty:
        raise ValueError("广东边界文件为空。")
    if boundary.crs is None:
        raise ValueError("广东边界缺少坐标参考信息，请检查 .prj 文件。")
    boundary = boundary.to_crs("EPSG:4326")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_PATTERN.format(
        date=args.date
    )

    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=FIGSIZE,
        constrained_layout=True,
    )

    for plot_index, axis in enumerate(axes.ravel()):
        row = plot_index // 3
        column = plot_index % 3

        specification = SELECTED_VARIABLES[plot_index]

        data = clean_and_mask(
            features[plot_index],
            mask,
        )

        image = axis.imshow(
            data,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
        )

        boundary.boundary.plot(
            ax=axis,
            linewidth=1.0,
        )

        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_title(
            str(specification["title"]),
            fontsize=SUBTITLE_SIZE,
            pad=10,
        )

        # 在子图左上角添加（a）～（i）编号。
        # 使用轴坐标定位，避免受到经纬度范围变化影响。
        if args.show_panel_labels:
            panel_letter = string.ascii_lowercase[
                plot_index
            ]
            axis.text(
                PANEL_LABEL_X,
                PANEL_LABEL_Y,
                f"({panel_letter})",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_SIZE,
                fontweight="bold",
                color="black",
                zorder=10,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                },
            )

        set_outer_ticks(
            axis=axis,
            extent=extent,
            show_longitude=(row == 2),
            show_latitude=(column == 0),
        )

        colorbar = figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.025,
        )
        colorbar.ax.tick_params(
            labelsize=COLORBAR_TICK_SIZE
        )

        unit = str(specification["unit"])
        if unit:
            colorbar.set_label(
                unit,
                fontsize=COLORBAR_LABEL_SIZE,
            )

    if "月均" in args.date:
        readable_date = (
            f"{args.date[0:4]}年"
            f"{int(args.date[4:6])}月月平均"
        )
    else:
        readable_date = (
            f"{args.date[0:4]}年"
            f"{int(args.date[4:6])}月"
            f"{int(args.date[6:8])}日"
        )
    # figure.suptitle(
    #     f"{readable_date}广东省典型驱动因子空间分布",
    #     fontsize=MAIN_TITLE_SIZE,
    #     fontweight="bold",
    # )

    figure.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("绘图完成。")
    print("输出文件：", output_path.resolve())
    print("选取通道：")
    for specification, channel in zip(
        SELECTED_VARIABLES,
        selected_channels,
    ):
        print(
            f"  {specification['title']}：通道 {channel}"
        )


if __name__ == "__main__":
    main()