#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化广东省每日 FIRMS 火灾置信度标签。

功能
----
- 读取 firms_conf_YYYYMMDD.npy
- 保留原始 FIRMS confidence，不进行二值化
- 叠加广东省边界
- 显示 xx°E / xx°N 形式的经纬度刻度
- 不显示 Longitude / Latitude 坐标说明
- 支持绘制单日或一段日期
- 每天输出一张 PNG，可选输出 PDF

依赖
----
conda install -c conda-forge numpy matplotlib geopandas shapely

运行示例
--------
绘制默认日期：
python visualize_firms_labels_guangdong.py

绘制指定日期：
python visualize_firms_labels_guangdong.py --date 20251001

绘制日期范围：
python visualize_firms_labels_guangdong.py \
    --start-date 20251001 \
    --end-date 20251015
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. 默认配置
# ============================================================
DATE = "20251203"

FIRMS_DIR = Path("./data/processed/firms_conf_1km")
FIRMS_PATTERN = "firms_conf_{date}.npy"

GRID_METADATA_PATH = Path("./data/processed/target_grid_metadata.json")
BOUNDARY_SHP = Path("./data/boundary/gd_boundary.shp")

OUTPUT_DIR = Path("./workspace/特征标签可视化/visualize_firms_labels_guangdong")

# FIRMS confidence 显示范围
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 100.0

SAVE_PNG = True
SAVE_PDF = False
DPI = 300

# 是否把 0 值隐藏。
# False：完整显示 0～100 的置信度栅格。
# True：仅突出 confidence > 0 的火点像元。
MASK_ZERO = False


# ============================================================
# 2. 参数与日期
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Guangdong FIRMS confidence labels."
    )

    parser.add_argument(
        "--date",
        default=None,
        help="Single date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Inclusive start date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date in YYYYMMDD format.",
    )
    parser.add_argument(
        "--firms-dir",
        default=str(FIRMS_DIR),
    )
    parser.add_argument(
        "--firms-pattern",
        default=FIRMS_PATTERN,
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
        "--output-dir",
        default=str(OUTPUT_DIR),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
    )
    parser.add_argument(
        "--mask-zero",
        action="store_true",
        default=MASK_ZERO,
        help="Hide zero-confidence pixels and show only fire pixels.",
    )

    return parser.parse_args()


def parse_yyyymmdd(text: str) -> datetime:
    try:
        return datetime.strptime(text, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(
            f"日期必须为 YYYYMMDD，当前为 {text!r}"
        ) from exc


def date_range(
    start_date: str,
    end_date: str,
) -> Iterable[str]:
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)

    if end < start:
        raise ValueError("end_date 不能早于 start_date。")

    current = start
    while current <= end:
        yield current.strftime("%Y%m%d")
        current += timedelta(days=1)


def choose_dates(args) -> list[str]:
    if args.date is not None:
        parse_yyyymmdd(args.date)
        return [args.date]

    if args.start_date is not None or args.end_date is not None:
        if args.start_date is None or args.end_date is None:
            raise ValueError(
                "--start-date 和 --end-date 必须同时提供。"
            )

        return list(
            date_range(args.start_date, args.end_date)
        )

    return [DATE]


# ============================================================
# 3. 坐标格式与元数据
# ============================================================
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


def load_grid_metadata(
    path: Path,
) -> Tuple[
    Tuple[float, float, float, float],
    int,
    int,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"找不到网格元数据：{path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    bounds = metadata["bounds"]

    extent = (
        float(bounds["left"]),
        float(bounds["right"]),
        float(bounds["bottom"]),
        float(bounds["top"]),
    )

    height = int(metadata["height"])
    width = int(metadata["width"])

    return extent, height, width


def make_ticks(
    minimum: float,
    maximum: float,
    count: int = 5,
) -> np.ndarray:
    return np.linspace(
        minimum,
        maximum,
        count,
    )


# ============================================================
# 4. 单日绘图
# ============================================================
def visualize_one_day(
    date_text: str,
    firms_path: Path,
    boundary: gpd.GeoDataFrame,
    extent: Tuple[float, float, float, float],
    expected_height: int,
    expected_width: int,
    output_dir: Path,
    dpi: int,
    mask_zero: bool,
) -> None:
    if not firms_path.exists():
        raise FileNotFoundError(
            f"找不到 FIRMS 标签：{firms_path}"
        )

    firms = np.load(firms_path)

    if firms.ndim == 3 and firms.shape[0] == 1:
        firms = firms[0]

    if firms.ndim != 2:
        raise ValueError(
            f"FIRMS 标签应为二维 (H, W)，当前为 {firms.shape}"
        )

    if firms.shape != (
        expected_height,
        expected_width,
    ):
        raise ValueError(
            f"{firms_path} shape={firms.shape}，"
            f"网格元数据要求 "
            f"{(expected_height, expected_width)}"
        )

    firms = np.asarray(
        firms,
        dtype=np.float32,
    )

    firms = np.nan_to_num(
        firms,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # 保留置信度，只把非法范围清理为 0。
    invalid = (
        (firms < CONFIDENCE_MIN)
        | (firms > CONFIDENCE_MAX)
    )
    firms[invalid] = 0.0

    display_data = firms
    if mask_zero:
        display_data = np.ma.masked_where(
            firms <= 0,
            firms,
        )

    fire_mask = firms > 0
    fire_pixel_count = int(
        np.count_nonzero(fire_mask)
    )

    if fire_pixel_count > 0:
        fire_confidence = firms[fire_mask]
        maximum_confidence = float(
            fire_confidence.max()
        )
        mean_confidence = float(
            fire_confidence.mean()
        )
    else:
        maximum_confidence = 0.0
        mean_confidence = 0.0

    left, right, bottom, top = extent

    figure, axis = plt.subplots(
        figsize=(9, 8),
        constrained_layout=True,
    )

    # print(display_data.shape)
    display_data = np.where(display_data==0, np.nan, display_data)
    # exit()
    image = axis.imshow(
        display_data,
        extent=extent,
        origin="upper",
        vmin=CONFIDENCE_MIN,
        vmax=CONFIDENCE_MAX,
        interpolation="nearest",
    )

    boundary.boundary.plot(
        ax=axis,
        linewidth=1.0,
    )

    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)

    axis.set_xticks(
        make_ticks(left, right, count=5)
    )
    axis.set_yticks(
        make_ticks(bottom, top, count=5)
    )

    axis.xaxis.set_major_formatter(
        FuncFormatter(format_longitude)
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(format_latitude)
    )

    # 不显示横纵坐标说明。
    axis.set_xlabel("")
    axis.set_ylabel("")

    axis.set_title(
        "Guangdong FIRMS Confidence - "
        f"{date_text}\n"
        f"Fire pixels: {fire_pixel_count}  |  "
        f"Mean confidence: {mean_confidence:.2f}  |  "
        f"Maximum confidence: {maximum_confidence:.2f}"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.045,
        pad=0.03,
    )
    colorbar.set_label(
        "FIRMS confidence"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_PNG:
        png_path = (
            output_dir
            / f"firms_conf_{date_text}.png"
        )
        figure.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
        )
        print("Saved PNG:", png_path)

    if SAVE_PDF:
        pdf_path = (
            output_dir
            / f"firms_conf_{date_text}.pdf"
        )
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )
        print("Saved PDF:", pdf_path)

    plt.close(figure)

    print(
        f"[CHECK] {date_text}: "
        f"shape={firms.shape}, "
        f"fire_pixels={fire_pixel_count}, "
        f"mean_confidence={mean_confidence:.4f}, "
        f"maximum_confidence={maximum_confidence:.4f}"
    )


# ============================================================
# 5. 主流程
# ============================================================
def main():
    args = parse_args()
    dates = choose_dates(args)

    firms_dir = Path(args.firms_dir)
    grid_metadata_path = Path(
        args.grid_metadata
    )
    boundary_shp = Path(
        args.boundary_shp
    )
    output_dir = Path(
        args.output_dir
    )

    extent, height, width = load_grid_metadata(
        grid_metadata_path
    )

    if not boundary_shp.exists():
        raise FileNotFoundError(
            f"找不到广东边界：{boundary_shp}"
        )

    boundary = gpd.read_file(
        boundary_shp
    )

    if boundary.empty:
        raise ValueError("广东边界文件为空。")

    if boundary.crs is None:
        raise ValueError(
            "广东边界缺少 CRS，请检查 gd_boundary.prj。"
        )

    boundary = boundary.to_crs(
        "EPSG:4326"
    )

    success_count = 0
    missing_count = 0

    for date_text in dates:
        firms_path = (
            firms_dir
            / args.firms_pattern.format(
                date=date_text
            )
        )

        if not firms_path.exists():
            print(
                "[MISSING]",
                firms_path,
            )
            missing_count += 1
            continue

        visualize_one_day(
            date_text=date_text,
            firms_path=firms_path,
            boundary=boundary,
            extent=extent,
            expected_height=height,
            expected_width=width,
            output_dir=output_dir,
            dpi=args.dpi,
            mask_zero=args.mask_zero,
        )

        success_count += 1

    print()
    print("[DONE]")
    print("Visualized:", success_count)
    print("Missing:", missing_count)
    print("Output directory:", output_dir.resolve())


if __name__ == "__main__":
    main()