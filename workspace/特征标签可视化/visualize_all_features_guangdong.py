#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化每日特征文件中的全部 72 个特征，并叠加广东边界与经纬度刻度。

功能
----
- 读取 feature_YYYYMMDD.npy，shape 应为 (72, H, W)
- 读取 feature_channel_metadata.json，自动生成每个通道标题
- 读取 target_grid_metadata.json，恢复空间范围
- 读取 gd_boundary.shp，叠加广东边界
- 将 72 个特征分多页绘制（默认每页 24 个子图）
- 经纬度刻度直接显示为 xx°E / xx°W / xx°N / xx°S
- 不显示横纵坐标说明（即不写 Longitude / Latitude）

依赖
----
conda install -c conda-forge numpy matplotlib geopandas shapely

示例
----
python visualize_all_features_guangdong.py --date 20251001

或者直接改脚本顶部默认参数后运行：
python visualize_all_features_guangdong.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. 默认配置
# ============================================================
DATE = "20251201"

FEATURE_DIR = Path("./data/processed/features_1km")
FEATURE_PATTERN = "feature_{date}.npy"

CHANNEL_METADATA_PATH = Path("./data/processed/feature_channel_metadata.json")
GRID_METADATA_PATH = Path("./data/processed/target_grid_metadata.json")
BOUNDARY_SHP = Path("./data/boundary/gd_boundary.shp")

OUTPUT_DIR = Path("./workspace/特征标签可视化/visualize_all_features_guangdong")

# 每页多少个子图：建议 24（6行4列）或 20（5行4列）
SUBPLOTS_PER_PAGE = 24
NROWS = 6
NCOLS = 4

# 是否同时保存 PDF 和 PNG
SAVE_PDF = False
SAVE_PNG = True

# PNG 分辨率
DPI = 200


# ============================================================
# 2. 工具函数
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize all Guangdong feature channels with boundary and lat/lon ticks."
    )
    parser.add_argument("--date", default=DATE, help="Date in YYYYMMDD format.")
    parser.add_argument("--feature-dir", default=str(FEATURE_DIR))
    parser.add_argument("--feature-pattern", default=FEATURE_PATTERN)
    parser.add_argument("--channel-metadata", default=str(CHANNEL_METADATA_PATH))
    parser.add_argument("--grid-metadata", default=str(GRID_METADATA_PATH))
    parser.add_argument("--boundary-shp", default=str(BOUNDARY_SHP))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--subplots-per-page", type=int, default=SUBPLOTS_PER_PAGE)
    parser.add_argument("--nrows", type=int, default=NROWS)
    parser.add_argument("--ncols", type=int, default=NCOLS)
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def format_lon(x, pos=None):
    if x > 180:
        x = x - 360
    if x < 0:
        return f"{abs(x):.1f}°W"
    return f"{x:.1f}°E"


def format_lat(y, pos=None):
    if y < 0:
        return f"{abs(y):.1f}°S"
    return f"{y:.1f}°N"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_extent(grid_metadata: Dict) -> Tuple[float, float, float, float]:
    bounds = grid_metadata["bounds"]
    left = float(bounds["left"])
    right = float(bounds["right"])
    bottom = float(bounds["bottom"])
    top = float(bounds["top"])
    return left, right, bottom, top


def build_channel_titles(channel_metadata: Dict) -> List[str]:
    order = channel_metadata["channel_order"]
    titles = []

    for item in order:
        channel = item["channel"]
        source = item["source"]
        variable = item["variable"]
        level = item.get("pressure_level_hpa", None)

        if source == "weather_upper" and level is not None:
            title = f"{channel:02d} | {variable} | {level} hPa"
        else:
            title = f"{channel:02d} | {variable}"

        titles.append(title)

    return titles


def choose_ticks(min_val: float, max_val: float, n: int = 4):
    if n < 2:
        return [0.5 * (min_val + max_val)]
    return np.linspace(min_val, max_val, n)


# ============================================================
# 3. 主流程
# ============================================================
def main():
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    feature_path = feature_dir / args.feature_pattern.format(date=args.date)
    channel_metadata_path = Path(args.channel_metadata)
    grid_metadata_path = Path(args.grid_metadata)
    boundary_shp = Path(args.boundary_shp)
    output_dir = Path(args.output_dir)

    if not feature_path.exists():
        raise FileNotFoundError(f"找不到特征文件：{feature_path}")
    if not channel_metadata_path.exists():
        raise FileNotFoundError(f"找不到通道元数据：{channel_metadata_path}")
    if not grid_metadata_path.exists():
        raise FileNotFoundError(f"找不到网格元数据：{grid_metadata_path}")
    if not boundary_shp.exists():
        raise FileNotFoundError(f"找不到广东边界：{boundary_shp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(feature_path)
    if features.ndim != 3:
        raise ValueError(f"特征数组应为三维 (C, H, W)，当前为 {features.shape}")

    num_channels, height, width = features.shape
    if num_channels != 72:
        print(f"[WARNING] 当前通道数为 {num_channels}，不是预期的 72，但仍继续绘制。")

    channel_metadata = load_json(channel_metadata_path)
    grid_metadata = load_json(grid_metadata_path)

    titles = build_channel_titles(channel_metadata)
    extent = build_extent(grid_metadata)

    boundary = gpd.read_file(boundary_shp)
    if boundary.crs is None:
        raise ValueError("广东边界缺少 CRS，请检查 .prj 文件。")
    boundary = boundary.to_crs("EPSG:4326")

    left, right, bottom, top = extent
    xticks = choose_ticks(left, right, n=4)
    yticks = choose_ticks(bottom, top, n=4)

    per_page = args.subplots_per_page
    nrows = args.nrows
    ncols = args.ncols

    if nrows * ncols < per_page:
        raise ValueError("nrows * ncols 必须不小于 subplots-per-page。")

    total_pages = math.ceil(num_channels / per_page)

    base_name = f"all_features_{args.date}"

    for page_idx in range(total_pages):
        start = page_idx * per_page
        end = min((page_idx + 1) * per_page, num_channels)
        count = end - start

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(16, 22),
            constrained_layout=True,
        )

        axes = np.atleast_1d(axes).ravel()

        for ax_i, channel_idx in enumerate(range(start, end)):
            ax = axes[ax_i]
            data = features[channel_idx]
            mask = np.load("./data/processed/guangdong_mask_1km.npy")
            data = np.where(mask==0, np.nan, data)
            # print(data.shape)
            # exit()

            im = ax.imshow(
                data,
                extent=extent,
                origin="upper",
                aspect="auto",
            )

            boundary.boundary.plot(ax=ax)

            ax.set_title(titles[channel_idx], fontsize=10)

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
            ax.yaxis.set_major_formatter(FuncFormatter(format_lat))

            # 不显示“Longitude / Latitude”说明
            ax.set_xlabel("")
            ax.set_ylabel("")

            # 每个子图都加自己的 colorbar，便于查看不同特征范围
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

        # 多余子图关闭
        for j in range(count, len(axes)):
            axes[j].axis("off")

        fig.suptitle(
            f"All Features - {args.date} (Page {page_idx + 1}/{total_pages})",
            fontsize=16,
        )

        if SAVE_PNG:
            png_path = output_dir / f"{base_name}_page{page_idx + 1}.png"
            fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
            print(f"Saved PNG: {png_path}")

        if SAVE_PDF:
            pdf_path = output_dir / f"{base_name}_page{page_idx + 1}.pdf"
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"Saved PDF: {pdf_path}")

        plt.close(fig)

    print("[DONE] Visualization completed.")


if __name__ == "__main__":
    main()