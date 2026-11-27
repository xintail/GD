#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
广东省 2026-12-01 至 2026-12-31 火灾预测交付脚本（简化 CSV 版）。

主要输出
--------
1. 每个火灾模型、每个日期分别导出：
   - 逐点概率 CSV：经度、纬度、地市、区县、预测概率；
   - 区县统计 CSV：地市、区县、最小概率、平均概率、最大概率；
   - reference_threshold.csv：模型参考阈值。

2. 每个气象变量、每个日期分别导出：
   - 逐点 CSV：经度、纬度、地市、区县、变量值；
   - 区县统计 CSV：地市、区县、最小值、平均值、最大值。
   单位直接写入文件名，不再写标题、表注或额外说明行。

3. 保留：
   - 8 个代表性气象变量的整月 GIF；
   - 每个模型的整月火灾概率图和分类图（7 列 × 5 行月历布局）。

性能调整
--------
- 不使用 openpyxl，不生成 XLSX、多 Sheet、颜色、筛选、冻结窗格或列宽；
- CSV 采用标准库 csv 流式写入；
- 气象缓存改为“每变量、每天一个未压缩 NPY”，避免反复解压整份 NPZ；
- 广东掩膜内未被区县多边形直接覆盖的少量边界像元，分配给最近的已分配像元，
  最终要求广东掩膜内所有像元均有区县归属。

依赖
----
conda install -c conda-forge \
    numpy scipy matplotlib pillow rasterio geopandas shapely pyogrio

运行
----
python export_guangdong_forecast_delivery.py
"""

from __future__ import annotations

import csv
import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter
from rasterio import Affine
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, map_coordinates
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

# 中文字体设置：优先使用 Windows 常见中文字体。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

try:
    from shapely import make_valid
except ImportError:
    from shapely.validation import make_valid



# =============================================================================
# 0. 配置
# =============================================================================
START_DATE = "2026-12-01"
END_DATE = "2026-12-31"

# 行政层级和边界
HIERARCHICAL_JSON = Path(
    "./data/Hierarchical_mapping/Hierarchical.json"
)
HIERARCHICAL_MAPPING_JSON = Path(
    "./data/Hierarchical_mapping/Hierarchical_mapping.json"
)

# 训练/预测共同使用的广东 1 km 网格
GRID_METADATA_PATH = Path(
    "./data/processed/target_grid_metadata.json"
)
GUANGDONG_MASK_PATH = Path(
    "./data/processed/guangdong_mask_1km.npy"
)

# 火灾预测结果（2026年12月报告以未校准随机森林为主结果，校准结果用于对照）
PREDICTION_ROOT = Path("./workspace/火灾预测/pangu_predictions")
CHECKPOINT_ROOT = Path("./checkpoint")
DELIVERY_ROOT = Path("./workspace/材料交付/export_guangdong_forecast_delivery")
# PREDICTION_ROOT = Path("./workspace/火灾预测/pangu_predictions_calibration")
# CHECKPOINT_ROOT = Path("./checkpoint_calibration")
# DELIVERY_ROOT = Path("./workspace/材料交付/export_guangdong_forecast_delivery_calibration")

MODEL_NAMES = [
    # "logistic_regression",
    # "sgd_logistic",
    "random_forest",
    # "extra_trees",
    # "ensemble",
]

# 全球 Pangu NPY
PANGU_NPY_DIR = Path("./Pangu-Weather-main/Pangu_npy")
UPPER_PATTERN = "input_upper_{date}.npy"
SURFACE_PATTERN = "input_surface_{date}.npy"

# 输出

# 是否覆盖已有输出
OVERWRITE = False

# 是否导出模型逐点 CSV、区县统计 CSV 和参考阈值。
EXPORT_MODEL_TABLES = True

# 是否导出逐点 CSV。整月逐点文件较大，报告只需要区县统计，默认关闭。
EXPORT_POINT_TABLES = False

# 是否准备 19 个气象变量的缓存。整月报告不使用气象 CSV/GIF，默认关闭。
PREPARE_WEATHER_CACHE = False

# 是否生成每个模型的整月火灾概率图和分类图。
GENERATE_FIRE_FIGURES = True

# 是否导出气象变量逐点/区县 CSV。
EXPORT_WEATHER_TABLES = False

# 是否生成 8 变量气象 GIF。
GENERATE_WEATHER_GIF = False

# 内部未压缩缓存；不是最终交付表。
CACHE_ROOT = DELIVERY_ROOT / "_cache"

# 图像和 GIF
FIGURE_DPI = 250
GIF_FPS = 1

# Pangu 全球网格
EXPECTED_SURFACE_SHAPE = (4, 721, 1440)
EXPECTED_UPPER_SHAPE = (5, 13, 721, 1440)
PANGU_PRESSURE_LEVELS = [
    1000, 925, 850, 700, 600, 500, 400,
    300, 250, 200, 150, 100, 50,
]

SURFACE_INDEX = {
    "msl": 0,
    "10u": 1,
    "10v": 2,
    "2t": 3,
}
UPPER_INDEX = {
    "z": 0,
    "q": 1,
    "t": 2,
    "u": 3,
    "v": 4,
}


@dataclass(frozen=True)
class WeatherSpec:
    key: str
    display_name: str
    source: str
    variable: str
    unit: str
    level_hpa: Optional[int] = None


WEATHER_SPECS: List[WeatherSpec] = [
    WeatherSpec("msl", "Mean sea-level pressure", "surface", "msl", "Pa"),
    WeatherSpec("10u", "10 m zonal wind", "surface", "10u", "m s-1"),
    WeatherSpec("10v", "10 m meridional wind", "surface", "10v", "m s-1"),
    WeatherSpec("2t", "2 m temperature", "surface", "2t", "K"),
]

for variable, display_name, unit in [
    ("z", "Geopotential", "m2 s-2"),
    ("q", "Specific humidity", "kg kg-1"),
    ("t", "Air temperature", "K"),
    ("u", "Zonal wind", "m s-1"),
    ("v", "Meridional wind", "m s-1"),
]:
    for level in [1000, 925, 850]:
        WEATHER_SPECS.append(
            WeatherSpec(
                key=f"{variable}_{level}hPa",
                display_name=f"{display_name} at {level} hPa",
                source="upper",
                variable=variable,
                unit=unit,
                level_hpa=level,
            )
        )

GIF_VARIABLE_KEYS = [
    "2t",
    "msl",
    "10u",
    "10v",
    "t_850hPa",
    "q_850hPa",
    "u_850hPa",
    "v_850hPa",
]


# =============================================================================
# 1. 通用工具
# =============================================================================
def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d")


def date_range(start: str, end: str) -> List[str]:
    current = parse_date(start)
    end_dt = parse_date(end)

    if end_dt < current:
        raise ValueError("END_DATE 不能早于 START_DATE。")

    dates: List[str] = []
    while current <= end_dt:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return dates


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def natural_tree_id(tree_id: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in tree_id.split("-"))


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


def apply_geo_ticks(axis, bounds: Tuple[float, float, float, float]) -> None:
    left, right, bottom, top = bounds
    axis.set_xlim(left, right)
    axis.set_ylim(bottom, top)
    axis.set_xticks(np.linspace(left, right, 4))
    axis.set_yticks(np.linspace(bottom, top, 4))
    axis.xaxis.set_major_formatter(FuncFormatter(format_longitude))
    axis.yaxis.set_major_formatter(FuncFormatter(format_latitude))
    axis.set_xlabel("")
    axis.set_ylabel("")


# =============================================================================
# 2. TopoJSON 解码与区县边界
# =============================================================================
def decode_topo_arc(topology: Mapping, arc_index: int) -> List[Tuple[float, float]]:
    reverse = arc_index < 0
    resolved_index = -arc_index - 1 if reverse else arc_index

    coordinates = [
        (float(point[0]), float(point[1]))
        for point in topology["arcs"][resolved_index]
    ]

    transform = topology.get("transform")
    if transform is not None:
        scale_x, scale_y = transform["scale"]
        translate_x, translate_y = transform["translate"]
        x = 0.0
        y = 0.0
        decoded: List[Tuple[float, float]] = []

        for delta_x, delta_y in coordinates:
            x += delta_x
            y += delta_y
            decoded.append(
                (
                    x * scale_x + translate_x,
                    y * scale_y + translate_y,
                )
            )
        coordinates = decoded

    if reverse:
        coordinates.reverse()

    return coordinates


def decode_topo_ring(topology: Mapping, arc_indexes: Sequence[int]) -> List[Tuple[float, float]]:
    ring: List[Tuple[float, float]] = []

    for arc_index in arc_indexes:
        arc = decode_topo_arc(topology, int(arc_index))
        if ring and arc and ring[-1] == arc[0]:
            arc = arc[1:]
        ring.extend(arc)

    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])

    return ring


def polygonal_parts(geometry) -> List[Polygon]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        parts: List[Polygon] = []
        for child in geometry.geoms:
            parts.extend(polygonal_parts(child))
        return parts
    return []


def topology_to_geometry(topology: Mapping):
    geometries = topology["objects"]["collection"]["geometries"]
    if len(geometries) != 1:
        raise ValueError("当前脚本要求每个层级节点只有一个几何对象。")

    topology_geometry = geometries[0]
    if topology_geometry["type"] != "MultiPolygon":
        raise ValueError(
            f"当前仅支持 MultiPolygon，实际为 {topology_geometry['type']}。"
        )

    polygons: List[Polygon] = []

    for polygon_arcs in topology_geometry["arcs"]:
        rings = [
            decode_topo_ring(topology, ring_arc_indexes)
            for ring_arc_indexes in polygon_arcs
        ]

        if not rings or len(rings[0]) < 4:
            continue

        polygon = Polygon(rings[0], rings[1:])
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        polygons.extend(polygonal_parts(polygon))

    if not polygons:
        raise ValueError("TopoJSON 没有产生有效多边形。")

    merged = unary_union(polygons)
    if not merged.is_valid:
        merged = make_valid(merged)

    parts = polygonal_parts(merged)
    if not parts:
        raise ValueError("修复后的几何不包含 Polygon。")

    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts)


def traverse_hierarchy(node: Mapping, depth: int = 0):
    yield depth, node
    children = node.get("children") or {}
    for child in children.values():
        yield from traverse_hierarchy(child, depth + 1)


def load_administrative_boundaries() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not HIERARCHICAL_JSON.exists():
        raise FileNotFoundError(f"找不到层级边界：{HIERARCHICAL_JSON}")
    if not HIERARCHICAL_MAPPING_JSON.exists():
        raise FileNotFoundError(f"找不到层级映射：{HIERARCHICAL_MAPPING_JSON}")

    root = load_json(HIERARCHICAL_JSON)
    mapping_rows = load_json(HIERARCHICAL_MAPPING_JSON)

    mapping_by_id = {
        str(item["treeID"]): item
        for item in mapping_rows
    }

    province_geometry = topology_to_geometry(root["payload"])
    province_gdf = gpd.GeoDataFrame(
        [{"name": root["name"], "geometry": province_geometry}],
        crs="EPSG:4326",
    )

    district_rows: List[Dict[str, object]] = []

    for depth, node in traverse_hierarchy(root):
        children = node.get("children") or {}

        # 常规区县：三级节点。
        is_district = depth == 2

        # 东莞市、中山市等没有区县子节点的地级市，作为区县等价统计单元，
        # 否则这些区域内的 1 km 像元无法归属。
        is_city_equivalent = depth == 1 and len(children) == 0

        if not is_district and not is_city_equivalent:
            continue

        tree_id = str(node["treeID"])
        mapping_item = mapping_by_id.get(tree_id)
        if mapping_item is None:
            raise KeyError(f"映射表缺少行政单元 treeID={tree_id}")

        if is_city_equivalent:
            city_tree_id = tree_id
            city_name = str(mapping_item["name"])
            district_name = str(mapping_item["name"])
            admin_level = "city_equivalent"
        else:
            city_tree_id = str(mapping_item["parent"])
            city_item = mapping_by_id.get(city_tree_id)
            if city_item is None:
                raise KeyError(f"映射表缺少地市 treeID={city_tree_id}")
            city_name = str(city_item["name"])
            district_name = str(mapping_item["name"])
            admin_level = "district"

        geometry = topology_to_geometry(node["payload"])

        district_rows.append(
            {
                "tree_id": tree_id,
                "city_tree_id": city_tree_id,
                "city": city_name,
                "district": district_name,
                "admin_level": admin_level,
                "geometry": geometry,
            }
        )

    district_rows.sort(key=lambda row: natural_tree_id(str(row["tree_id"])))

    for district_id, row in enumerate(district_rows, start=1):
        row["district_id"] = district_id

    district_gdf = gpd.GeoDataFrame(
        district_rows,
        geometry="geometry",
        crs="EPSG:4326",
    )

    if len(district_gdf) != 124:
        print(
            f"[WARNING] 区县/等价统计单元数量为 {len(district_gdf)}，"
            "当前层级文件预期为 122 个区县 + 东莞市和中山市 2 个等价单元。"
        )

    return province_gdf, district_gdf


# =============================================================================
# 3. 目标网格和区县编号栅格
# =============================================================================
def load_target_grid() -> Tuple[rasterio.crs.CRS, Affine, int, int, Tuple[float, float, float, float]]:
    metadata = load_json(GRID_METADATA_PATH)

    crs = rasterio.crs.CRS.from_string(metadata["crs"])
    transform = Affine(*map(float, metadata["transform"]))
    height = int(metadata["height"])
    width = int(metadata["width"])

    bounds_data = metadata["bounds"]
    bounds = (
        float(bounds_data["left"]),
        float(bounds_data["right"]),
        float(bounds_data["bottom"]),
        float(bounds_data["top"]),
    )

    return crs, transform, height, width, bounds


def target_lon_lat(transform: Affine, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((height, width), dtype=np.float64)

    longitude = (
        transform.c
        + (cols + 0.5) * transform.a
        + (rows + 0.5) * transform.b
    )
    latitude = (
        transform.f
        + (cols + 0.5) * transform.d
        + (rows + 0.5) * transform.e
    )

    return longitude, latitude


def build_district_grid(
    district_gdf: gpd.GeoDataFrame,
    target_crs,
    transform: Affine,
    height: int,
    width: int,
    guangdong_mask: np.ndarray,
) -> np.ndarray:
    """
    先按像元中心栅格化区县边界，再将省级掩膜内的边界缝隙像元
    分配给最近的已分配像元，保证广东掩膜内 100% 有区县归属。
    """
    reprojected = district_gdf.to_crs(target_crs)

    shapes = [
        (geometry, int(district_id))
        for geometry, district_id in zip(
            reprojected.geometry,
            reprojected["district_id"],
        )
        if geometry is not None and not geometry.is_empty
    ]

    district_grid = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=np.int16,
    )
    district_grid[~guangdong_mask] = 0

    unassigned_mask = guangdong_mask & (district_grid == 0)
    unassigned_count = int(np.count_nonzero(unassigned_mask))

    if unassigned_count:
        print(
            f"[DISTRICT FILL] {unassigned_count} 个边界缝隙像元按最近区县补齐。"
        )

        # distance_transform_edt 对 True 单元寻找最近的 False 单元。
        # district_grid == 0 为 True；已有区县编号的位置为 False。
        nearest_indices = distance_transform_edt(
            district_grid == 0,
            return_distances=False,
            return_indices=True,
        )

        nearest_rows = nearest_indices[0][unassigned_mask]
        nearest_cols = nearest_indices[1][unassigned_mask]
        district_grid[unassigned_mask] = district_grid[
            nearest_rows,
            nearest_cols,
        ]

    district_grid[~guangdong_mask] = 0

    remaining = int(
        np.count_nonzero(guangdong_mask & (district_grid == 0))
    )
    if remaining:
        raise RuntimeError(
            f"区县补齐后仍有 {remaining} 个广东像元没有区县归属。"
        )

    print(
        f"[DISTRICT CHECK] 广东掩膜内 "
        f"{int(np.count_nonzero(guangdong_mask))} 个像元已全部分配。"
    )
    return district_grid


@dataclass
class PointMetadata:
    flat_indices: np.ndarray
    longitude: np.ndarray
    latitude: np.ndarray
    district_ids: np.ndarray
    city: np.ndarray
    district: np.ndarray
    positions_by_district: Dict[int, np.ndarray]


def build_point_metadata(
    district_grid: np.ndarray,
    district_gdf: gpd.GeoDataFrame,
    transform: Affine,
) -> PointMetadata:
    height, width = district_grid.shape
    longitude_grid, latitude_grid = target_lon_lat(transform, height, width)

    valid_flat_indices = np.flatnonzero(district_grid.ravel() > 0)
    district_ids = district_grid.ravel()[valid_flat_indices].astype(np.int16)
    longitude = longitude_grid.ravel()[valid_flat_indices].astype(np.float64)
    latitude = latitude_grid.ravel()[valid_flat_indices].astype(np.float64)

    city_by_id = {
        int(row.district_id): str(row.city)
        for row in district_gdf.itertuples()
    }
    district_by_id = {
        int(row.district_id): str(row.district)
        for row in district_gdf.itertuples()
    }

    city = np.asarray(
        [city_by_id[int(value)] for value in district_ids],
        dtype=object,
    )
    district = np.asarray(
        [district_by_id[int(value)] for value in district_ids],
        dtype=object,
    )

    # 区县分组优先，其次纬度从北向南、经度从西向东。
    order = np.lexsort((longitude, -latitude, district_ids))

    flat_indices = valid_flat_indices[order]
    longitude = longitude[order]
    latitude = latitude[order]
    district_ids = district_ids[order]
    city = city[order]
    district = district[order]

    positions_by_district: Dict[int, np.ndarray] = {}
    for district_id in district_gdf["district_id"].astype(int):
        positions_by_district[int(district_id)] = np.flatnonzero(
            district_ids == int(district_id)
        )

    return PointMetadata(
        flat_indices=flat_indices,
        longitude=longitude,
        latitude=latitude,
        district_ids=district_ids,
        city=city,
        district=district,
        positions_by_district=positions_by_district,
    )


# =============================================================================
# 4. 简单 CSV 写入
# =============================================================================
def write_csv(
    path: Path,
    headers: Sequence[str],
    rows: Iterable[Sequence],
) -> Path:
    """使用大缓冲区流式写入最简单的 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not OVERWRITE:
        print("[CSV SKIP]", path)
        return path

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
        buffering=1024 * 1024,
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)

    print("[CSV]", path)
    return path


def unit_slug(unit: str) -> str:
    """把单位转换成适合文件名的简短形式。"""
    return (
        str(unit)
        .strip()
        .replace(" ", "_")
        .replace("/", "_per_")
        .replace("²", "2")
        .replace("⁻", "-")
    )


# =============================================================================
# 5. 模型概率读取、阈值和统计
# =============================================================================
def probability_path(model_name: str, date_text: str) -> Path:
    model_dir = PREDICTION_ROOT / model_name
    if model_name == "ensemble":
        return model_dir / f"mean_probability_{date_text}.npy"
    return model_dir / f"fire_probability_{date_text}.npy"


def load_probability_map(model_name: str, date_text: str, expected_shape: Tuple[int, int]) -> np.ndarray:
    path = probability_path(model_name, date_text)
    if not path.exists():
        raise FileNotFoundError(f"缺少模型概率：{path}")

    array = np.load(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]

    if array.shape != expected_shape:
        raise ValueError(
            f"{path} shape={array.shape}，目标网格={expected_shape}"
        )

    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(array, 0.0, 1.0)


def model_threshold_rows(model_name: str):
    """返回最小化的阈值 CSV 行；不写说明文字。"""
    if model_name == "ensemble":
        found = False
        for child_model in MODEL_NAMES:
            if child_model == "ensemble":
                continue
            path = CHECKPOINT_ROOT / child_model / "threshold.json"
            if path.exists():
                found = True
                yield (
                    child_model,
                    f"{float(load_json(path)['threshold']):.8f}",
                )
        if not found:
            yield ("ensemble", "")
        return

    threshold_path = CHECKPOINT_ROOT / model_name / "threshold.json"
    if threshold_path.exists():
        threshold = float(load_json(threshold_path)["threshold"])
        yield (model_name, f"{threshold:.8f}")
    else:
        yield (model_name, "")



def values_for_points(array_2d: np.ndarray, point_metadata: PointMetadata) -> np.ndarray:
    return array_2d.ravel()[point_metadata.flat_indices].astype(np.float64)


def point_rows(
    point_metadata: PointMetadata,
    values: np.ndarray,
    value_decimals: int = 8,
):
    value_format = f".{{}}f".format(value_decimals)
    for lon, lat, city, district, value in zip(
        point_metadata.longitude,
        point_metadata.latitude,
        point_metadata.city,
        point_metadata.district,
        values,
    ):
        yield (
            f"{float(lon):.6f}",
            f"{float(lat):.6f}",
            str(city),
            str(district),
            format(float(value), value_format),
        )


def district_statistics(
    values: np.ndarray,
    point_metadata: PointMetadata,
    district_gdf: gpd.GeoDataFrame,
) -> List[Tuple[str, str, float, float, float]]:
    """只保留地市、区县、最小值、平均值和最大值。"""
    rows: List[Tuple[str, str, float, float, float]] = []

    district_lookup = {
        int(row.district_id): (str(row.city), str(row.district))
        for row in district_gdf.itertuples()
    }

    for district_id in district_gdf["district_id"].astype(int):
        positions = point_metadata.positions_by_district[int(district_id)]
        if positions.size == 0:
            continue

        district_values = values[positions]
        city, district = district_lookup[int(district_id)]
        rows.append(
            (
                city,
                district,
                float(np.min(district_values)),
                float(np.mean(district_values)),
                float(np.max(district_values)),
            )
        )

    return rows


def formatted_statistics_rows(
    statistics: Sequence[Tuple[str, str, float, float, float]],
    decimals: int = 8,
):
    value_format = f".{{}}f".format(decimals)
    for city, district, minimum, mean, maximum in statistics:
        yield (
            city,
            district,
            format(minimum, value_format),
            format(mean, value_format),
            format(maximum, value_format),
        )


def export_model_tables(
    model_name: str,
    dates: Sequence[str],
    target_shape: Tuple[int, int],
    point_metadata: PointMetadata,
    district_gdf: gpd.GeoDataFrame,
) -> List[Path]:
    model_root = DELIVERY_ROOT / "models" / model_name
    point_root = model_root / "points"
    stats_root = model_root / "district_statistics"

    output_paths: List[Path] = []

    threshold_path = write_csv(
        model_root / "reference_threshold.csv",
        ["model", "reference_threshold"],
        model_threshold_rows(model_name),
    )
    output_paths.append(threshold_path)

    for date_text in dates:
        probability = load_probability_map(
            model_name,
            date_text,
            target_shape,
        )
        values = values_for_points(
            probability,
            point_metadata,
        )
        statistics = district_statistics(
            values,
            point_metadata,
            district_gdf,
        )

        point_path = None
        if EXPORT_POINT_TABLES:
            point_path = write_csv(
                point_root / f"fire_probability_points_{date_text}.csv",
                ["经度", "纬度", "地市", "区县", "预测概率"],
                point_rows(point_metadata, values, value_decimals=8),
            )
        stats_path = write_csv(
            stats_root / f"fire_probability_district_statistics_{date_text}.csv",
            ["地市", "区县", "最小概率", "平均概率", "最大概率"],
            formatted_statistics_rows(statistics, decimals=8),
        )
        if point_path is not None:
            output_paths.append(point_path)
        output_paths.append(stats_path)

        del probability
        del values
        gc.collect()

    return output_paths


# =============================================================================
# 6. Pangu 气象匹配和缓存
# =============================================================================
def pangu_source_indices(
    transform: Affine,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    longitude, latitude = target_lon_lat(transform, height, width)
    longitude = np.mod(longitude, 360.0)

    source_rows = (90.0 - latitude) / 0.25
    source_cols = longitude / 0.25
    return source_rows, source_cols


def load_pangu_arrays(date_text: str) -> Tuple[np.ndarray, np.ndarray]:
    upper_path = PANGU_NPY_DIR / UPPER_PATTERN.format(date=date_text)
    surface_path = PANGU_NPY_DIR / SURFACE_PATTERN.format(date=date_text)

    if not upper_path.exists():
        raise FileNotFoundError(f"缺少 Pangu upper：{upper_path}")
    if not surface_path.exists():
        raise FileNotFoundError(f"缺少 Pangu surface：{surface_path}")

    upper = np.load(upper_path, mmap_mode="r")
    surface = np.load(surface_path, mmap_mode="r")

    if upper.shape != EXPECTED_UPPER_SHAPE:
        raise ValueError(
            f"{upper_path} shape={upper.shape}，期望 {EXPECTED_UPPER_SHAPE}"
        )
    if surface.shape != EXPECTED_SURFACE_SHAPE:
        raise ValueError(
            f"{surface_path} shape={surface.shape}，期望 {EXPECTED_SURFACE_SHAPE}"
        )

    return upper, surface


def resample_weather_channel(
    source: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
) -> np.ndarray:
    result = map_coordinates(
        np.asarray(source, dtype=np.float32),
        [source_rows, source_cols],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return result.astype(np.float32, copy=False)


def extract_weather_source(spec: WeatherSpec, upper: np.ndarray, surface: np.ndarray) -> np.ndarray:
    if spec.source == "surface":
        return surface[SURFACE_INDEX[spec.variable]]

    if spec.source == "upper":
        if spec.level_hpa not in PANGU_PRESSURE_LEVELS:
            raise ValueError(f"Pangu 不包含 {spec.level_hpa} hPa")
        level_index = PANGU_PRESSURE_LEVELS.index(int(spec.level_hpa))
        return upper[UPPER_INDEX[spec.variable], level_index]

    raise ValueError(f"未知气象来源：{spec.source}")


def weather_cache_path(date_text: str, key: str) -> Path:
    return (
        CACHE_ROOT
        / "weather"
        / key
        / f"{key}_{date_text}.npy"
    )


def prepare_weather_cache(
    dates: Sequence[str],
    source_rows: np.ndarray,
    source_cols: np.ndarray,
    guangdong_mask: np.ndarray,
) -> None:
    """
    每个变量、每天保存为独立未压缩 NPY。
    读取某个变量时无需解压同一天的其他 18 个变量。
    """
    for date_text in dates:
        missing_specs = [
            spec
            for spec in WEATHER_SPECS
            if OVERWRITE
            or not weather_cache_path(date_text, spec.key).exists()
        ]

        if not missing_specs:
            print("[WEATHER CACHE SKIP]", date_text)
            continue

        print("[WEATHER CACHE]", date_text)
        upper, surface = load_pangu_arrays(date_text)

        for spec in missing_specs:
            source = extract_weather_source(spec, upper, surface)
            matched = resample_weather_channel(
                source,
                source_rows,
                source_cols,
            )
            matched[~guangdong_mask] = 0.0

            path = weather_cache_path(date_text, spec.key)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matched, allow_pickle=False)
            print("   ", path)

            del matched

        del upper
        del surface
        gc.collect()


def load_cached_weather(date_text: str, key: str) -> np.ndarray:
    path = weather_cache_path(date_text, key)
    if not path.exists():
        raise FileNotFoundError(f"缺少气象缓存：{path}")
    return np.load(path, mmap_mode="r")


def export_weather_tables(
    dates: Sequence[str],
    point_metadata: PointMetadata,
    district_gdf: gpd.GeoDataFrame,
) -> List[Path]:
    output_paths: List[Path] = []

    for spec in WEATHER_SPECS:
        unit_name = unit_slug(spec.unit)
        print("[WEATHER CSV]", spec.key)

        variable_root = (
            DELIVERY_ROOT
            / "weather"
            / "tables"
            / spec.key
        )
        point_root = variable_root / "points"
        stats_root = variable_root / "district_statistics"

        for date_text in dates:
            weather_map = load_cached_weather(
                date_text,
                spec.key,
            )
            values = values_for_points(
                weather_map,
                point_metadata,
            )
            statistics = district_statistics(
                values,
                point_metadata,
                district_gdf,
            )

            point_path = write_csv(
                point_root
                / f"{spec.key}_{unit_name}_points_{date_text}.csv",
                ["经度", "纬度", "地市", "区县", spec.key],
                point_rows(point_metadata, values, value_decimals=8),
            )
            stats_path = write_csv(
                stats_root
                / f"{spec.key}_{unit_name}_district_statistics_{date_text}.csv",
                ["地市", "区县", "最小值", "平均值", "最大值"],
                formatted_statistics_rows(statistics, decimals=8),
            )
            output_paths.extend([point_path, stats_path])

            del weather_map
            del values
            gc.collect()

    return output_paths


# =============================================================================
# 7. 气象 8 变量 GIF
# =============================================================================
def robust_limits(arrays: Sequence[np.ndarray], mask: np.ndarray, signed: bool) -> Tuple[float, float]:
    samples = np.concatenate(
        [np.asarray(array[mask], dtype=np.float64) for array in arrays]
    )
    samples = samples[np.isfinite(samples)]

    if samples.size == 0:
        return 0.0, 1.0

    lower = float(np.percentile(samples, 2))
    upper = float(np.percentile(samples, 98))

    if signed:
        maximum = max(abs(lower), abs(upper))
        if math.isclose(maximum, 0.0):
            maximum = 1.0
        return -maximum, maximum

    if math.isclose(lower, upper):
        upper = lower + 1.0

    return lower, upper


def create_weather_gif(
    dates: Sequence[str],
    guangdong_mask: np.ndarray,
    province_gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float],
) -> Path:
    spec_by_key = {spec.key: spec for spec in WEATHER_SPECS}

    missing = [key for key in GIF_VARIABLE_KEYS if key not in spec_by_key]
    if missing:
        raise KeyError(f"GIF 变量不在 WEATHER_SPECS：{missing}")

    data: Dict[str, List[np.ndarray]] = {
        key: [load_cached_weather(date_text, key) for date_text in dates]
        for key in GIF_VARIABLE_KEYS
    }

    limits: Dict[str, Tuple[float, float]] = {}
    for key in GIF_VARIABLE_KEYS:
        signed = key.startswith("10u") or key.startswith("10v") or key.startswith("u_") or key.startswith("v_")
        limits[key] = robust_limits(data[key], guangdong_mask, signed)

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(18, 9),
        constrained_layout=True,
    )
    axes = axes.ravel()

    images = []
    for axis, key in zip(axes, GIF_VARIABLE_KEYS):
        spec = spec_by_key[key]
        initial = np.ma.masked_where(~guangdong_mask, data[key][0])
        vmin, vmax = limits[key]

        image = axis.imshow(
            initial,
            extent=bounds,
            origin="upper",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        province_gdf.boundary.plot(ax=axis, linewidth=0.8)
        apply_geo_ticks(axis, bounds)
        axis.set_title(f"{spec.key} ({spec.unit})")
        figure.colorbar(image, ax=axis, fraction=0.045, pad=0.02)
        images.append(image)

    title = figure.suptitle(
        f"Pangu weather over Guangdong — {dates[0]}",
        fontsize=19,
    )

    def update(frame_index: int):
        date_text = dates[frame_index]
        for image, key in zip(images, GIF_VARIABLE_KEYS):
            image.set_data(
                np.ma.masked_where(
                    ~guangdong_mask,
                    data[key][frame_index],
                )
            )
        title.set_text(f"Pangu weather over Guangdong — {date_text}")
        return [*images, title]

    animation = FuncAnimation(
        figure,
        update,
        frames=len(dates),
        interval=1000,
        blit=False,
    )

    output_path = (
        DELIVERY_ROOT
        / "weather"
        / "figures"
        / "weather_8variables_20261001_20261007.gif"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=GIF_FPS))
    plt.close(figure)

    print("[GIF]", output_path)
    return output_path


# =============================================================================
# 8. 整月火灾概率图
# =============================================================================
def district_mean_mapping(
    values: np.ndarray,
    point_metadata: PointMetadata,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for district_id, positions in point_metadata.positions_by_district.items():
        if positions.size:
            result[int(district_id)] = float(np.mean(values[positions]))
    return result



def month_calendar_positions(
    dates: Sequence[str],
) -> Tuple[Dict[str, Tuple[int, int]], int, int]:
    """
    将连续日期放入 7 列月历网格。

    星期一为第 0 列，返回每个日期的 (行, 列)、行数和列数。
    """
    if not dates:
        raise ValueError("日期列表不能为空。")

    first_date = datetime.strptime(dates[0], "%Y%m%d")
    offset = first_date.weekday()

    positions: Dict[str, Tuple[int, int]] = {}
    max_row = 0

    for index, date_text in enumerate(dates):
        calendar_index = offset + index
        row = calendar_index // 7
        column = calendar_index % 7
        positions[date_text] = (row, column)
        max_row = max(max_row, row)

    return positions, max_row + 1, 7


def create_fire_probability_figure(
    model_name: str,
    dates: Sequence[str],
    target_shape: Tuple[int, int],
    guangdong_mask: np.ndarray,
    point_metadata: PointMetadata,
    district_gdf: gpd.GeoDataFrame,
    province_gdf: gpd.GeoDataFrame,
    bounds: Tuple[float, float, float, float],
) -> List[Path]:
    """
    分别生成两张整月预测图：

    1. 火灾概率图：
       - 7 列 × 5 行月历布局；
       - 每个子图对应一个日期；
       - 保留原脚本的区县平均概率绘制方式。

    2. 阈值分类图：
       - 7 列 × 5 行月历布局；
       - 每个子图对应一个日期。

    普通模型：
        probability >= 模型最佳 F1 参考概率。

    集成模型：
        各基础模型分别按自身参考概率分类，
        达到严格多数票时判定为火灾。
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if not dates:
        raise ValueError("整月预测图要求至少包含1天。")

    model_display_names = {
        "logistic_regression": "逻辑回归",
        "sgd_logistic": "随机梯度下降逻辑分类器",
        "random_forest": "随机森林",
        "extra_trees": "极端随机树",
        "ensemble": "集成模型",
    }
    model_display_name = model_display_names.get(
        model_name,
        model_name,
    )

    def format_chinese_date(date_text: str) -> str:
        date_value = datetime.strptime(date_text, "%Y%m%d")
        return (
            f"{date_value.year}."
            f"{date_value.month}."
            f"{date_value.day:02d}"
        )

    def enlarge_axis_text(axis) -> None:
        axis.tick_params(
            axis="both",
            which="major",
            labelsize=13,
        )

    def set_probability_colorbar_format(
        colorbar,
        maximum_probability: float,
    ) -> None:
        """
        当概率整体较小时，自动使用乘数形式显示颜色条。

        例如原始范围约为 0～0.0035 时，颜色条刻度显示为
        0～3.5，标题写为“预测火灾概率（×10⁻³）”。
        该操作只改变刻度显示，不改变原始概率和颜色映射。
        """
        if (
            np.isfinite(maximum_probability)
            and maximum_probability > 0
            and maximum_probability < 0.01
        ):
            exponent = (
                int(
                    math.floor(
                        math.log10(maximum_probability) / 3.0
                    )
                )
                * 3
            )
            scale = 10.0 ** (-exponent)

            colorbar.locator = MaxNLocator(nbins=5)
            colorbar.formatter = FuncFormatter(
                lambda value, position: (
                    f"{value * scale:.2f}"
                    .rstrip("0")
                    .rstrip(".")
                )
            )
            colorbar.update_ticks()
            colorbar.ax.xaxis.get_offset_text().set_visible(False)
            colorbar.set_ticks(
                np.linspace(
                    0.0,
                    maximum_probability,
                    5,
                )
            )
            colorbar.set_label(
                # f"预测火灾概率（×10$^{exponent_to_superscript(exponent)}$）",
                # "预测火灾概率（×10$^{-6}$）",
                "Fire Probability(×10$^{-6}$)",
                fontsize=20,
                labelpad=10,
            )
        else:
            colorbar.locator = MaxNLocator(nbins=5)
            colorbar.formatter = FuncFormatter(
                lambda value, position: (
                    f"{value:.3f}"
                    .rstrip("0")
                    .rstrip(".")
                )
            )
            colorbar.update_ticks()
            colorbar.ax.xaxis.get_offset_text().set_visible(False)
            colorbar.set_label(
                "预测火灾概率",
                fontsize=20,
                labelpad=10,
            )

        colorbar.ax.tick_params(
            axis="x",
            labelsize=18,
        )

    def exponent_to_superscript(exponent: int) -> str:
        translation = str.maketrans(
            {
                "-": "-",
                "0": "0",
                "1": "1",
                "2": "2",
                "3": "3",
                "4": "4",
                "5": "5",
                "6": "6",
                "7": "7",
                "8": "8",
                "9": "9",
            }
        )
        return str(exponent).translate(translation)

    # -------------------------------------------------------------------------
    # 1. 读取整月概率图
    # -------------------------------------------------------------------------
    probability_maps = [
        load_probability_map(
            model_name,
            date_text,
            target_shape,
        )
        for date_text in dates
    ]

    vmax = max(
        float(np.nanmax(probability))
        for probability in probability_maps
    )
    # 防止所有概率均为0时出现无效归一化范围。
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    vmax=0.162
    
    # -------------------------------------------------------------------------
    # 2. 生成阈值分类图
    # -------------------------------------------------------------------------
    binary_maps: List[np.ndarray] = []

    if model_name != "ensemble":
        threshold_path = (
            CHECKPOINT_ROOT
            / model_name
            / "threshold.json"
        )

        if not threshold_path.exists():
            raise FileNotFoundError(
                f"找不到模型参考概率文件：{threshold_path}"
            )

        threshold_payload = load_json(threshold_path)
        threshold = float(
            threshold_payload["threshold"]
        )

        for probability in probability_maps:
            binary = (
                probability >= threshold
            ).astype(np.uint8)

            binary[~guangdong_mask] = 0
            binary_maps.append(binary)

        classification_description = (
            f"参考概率={threshold:.6f}"
        )

    else:
        child_models = [
            name
            for name in MODEL_NAMES
            if name != "ensemble"
        ]

        if not child_models:
            raise ValueError(
                "MODEL_NAMES中没有集成模型所需的基础模型。"
            )

        required_votes = (
            len(child_models) // 2 + 1
        )

        child_thresholds: Dict[str, float] = {}

        for child_model in child_models:
            threshold_path = (
                CHECKPOINT_ROOT
                / child_model
                / "threshold.json"
            )

            if not threshold_path.exists():
                raise FileNotFoundError(
                    f"找不到子模型参考概率文件："
                    f"{threshold_path}"
                )

            child_thresholds[child_model] = float(
                load_json(threshold_path)["threshold"]
            )

        for date_text in dates:
            vote_count = np.zeros(
                target_shape,
                dtype=np.uint8,
            )

            for child_model in child_models:
                child_probability = (
                    load_probability_map(
                        child_model,
                        date_text,
                        target_shape,
                    )
                )

                vote_count += (
                    child_probability
                    >= child_thresholds[child_model]
                ).astype(np.uint8)

                del child_probability

            binary = (
                vote_count >= required_votes
            ).astype(np.uint8)

            binary[~guangdong_mask] = 0
            binary_maps.append(binary)

        classification_description = (
            f"严格多数投票："
            f"{required_votes}/{len(child_models)}票"
        )

    figure_dir = (
        DELIVERY_ROOT
        / "models"
        / model_name
        / "figures"
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths: List[Path] = []

    # -------------------------------------------------------------------------
    # 3. 绘制整月火灾概率图：7列 × 5行月历布局
    # -------------------------------------------------------------------------
    positions, nrows, ncols = month_calendar_positions(dates)

    probability_figure, probability_axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        constrained_layout=True,
    )
    probability_axes = np.asarray(probability_axes).reshape(nrows, ncols)
    used_probability_axes = []

    for axis in probability_axes.ravel():
        axis.axis("off")

    for day_index, (date_text, probability) in enumerate(
        zip(dates, probability_maps)
    ):
        row, column = positions[date_text]
        axis = probability_axes[row, column]
        axis.axis("on")
        used_probability_axes.append(axis)

        values = values_for_points(
            probability,
            point_metadata,
        )
        values = np.where(values<0.001, np.nan, values)

        mean_by_id = district_mean_mapping(
            values,
            point_metadata,
        )

        plot_gdf = district_gdf.copy()
        plot_gdf["mean_probability"] = (
            plot_gdf["district_id"].map(
                mean_by_id
            )
        )

        plot_gdf.plot(
            column="mean_probability",
            ax=axis,
            vmin=0.0,
            vmax=vmax,
            linewidth=0.45,
            edgecolor="black",
            missing_kwds={"color": "white"},
        )

        district_gdf.boundary.plot(
            ax=axis,
            linewidth=0.45,
        )

        apply_geo_ticks(
            axis,
            bounds,
        )
        enlarge_axis_text(axis)

        axis.set_title(
            format_chinese_date(date_text),
            fontsize=18,
            pad=8,
        )

        del values
        del mean_by_id
        del plot_gdf

    probability_mappable = ScalarMappable(
        norm=Normalize(
            vmin=0.0,
            vmax=vmax,
        ),
        cmap="viridis",
    )
    probability_mappable.set_array([])

    probability_colorbar = probability_figure.colorbar(
        probability_mappable,
        ax=used_probability_axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.045,
        aspect=60,
    )

    set_probability_colorbar_format(
        probability_colorbar,
        maximum_probability=vmax,
    )

    # probability_figure.suptitle(
    #     f"广东省整月火灾预测概率——{model_display_name}",
    #     fontsize=22,
    #     fontweight="bold",
    # )

    probability_output_path = (
        figure_dir
        / f"fire_probability_month_{model_name}.png"
    )
    pdf_probability_output_path = (
        figure_dir
        / f"fire_probability_month_{model_name}.pdf"
    )

    probability_figure.savefig(
        probability_output_path,
        dpi=FIGURE_DPI,
        bbox_inches=None,
    )
    probability_figure.savefig(
        pdf_probability_output_path,
        dpi=FIGURE_DPI,
        bbox_inches=None,
    )
    plt.close(probability_figure)

    print("[图像]", probability_output_path)
    output_paths.append(probability_output_path)

    # -------------------------------------------------------------------------
    # 4. 绘制整月阈值分类图：7列 × 5行月历布局
    # -------------------------------------------------------------------------
    binary_cmap = ListedColormap(
        [
            "#F2F2F2",  # 0：非火灾
            "#D73027",  # 1：火灾
        ]
    )

    binary_norm = BoundaryNorm(
        [-0.5, 0.5, 1.5],
        binary_cmap.N,
    )

    classification_figure, classification_axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        constrained_layout=True,
    )
    classification_axes = np.asarray(classification_axes).reshape(
        nrows,
        ncols,
    )
    used_classification_axes = []

    for axis in classification_axes.ravel():
        axis.axis("off")

    binary_mappable = None

    for day_index, (date_text, binary) in enumerate(
        zip(dates, binary_maps)
    ):
        row, column = positions[date_text]
        axis = classification_axes[row, column]
        axis.axis("on")
        used_classification_axes.append(axis)

        binary_mappable = axis.imshow(
            np.ma.masked_where(
                ~guangdong_mask,
                binary,
            ),
            extent=bounds,
            origin="upper",
            cmap=binary_cmap,
            norm=binary_norm,
            interpolation="nearest",
            aspect="auto",
        )

        province_gdf.boundary.plot(
            ax=axis,
            linewidth=1.0,
        )

        # 保留原脚本逻辑：前两天额外显示区县边界。
        if day_index < 2:
            district_gdf.boundary.plot(
                ax=axis,
                linewidth=0.3,
            )

        apply_geo_ticks(
            axis,
            bounds,
        )
        enlarge_axis_text(axis)

        fire_pixel_count = int(
            np.count_nonzero(
                binary[guangdong_mask] == 1
            )
        )

        axis.set_title(
            f"{format_chinese_date(date_text)}",
            # f"预测火灾像元数：{fire_pixel_count}",
            fontsize=18,
            fontweight="bold",
            pad=8,
        )

    classification_colorbar = classification_figure.colorbar(
        binary_mappable,
        ax=used_classification_axes,
        ticks=[0, 1],
        orientation="horizontal",
        fraction=0.035,
        pad=0.045,
        aspect=60,
    )

    classification_colorbar.ax.set_xticklabels(
        [
            "Non-fire",
            "Fire",
        ]
    )
    classification_colorbar.set_label(
        "Threshold Classification Result",
        fontsize=20,
        labelpad=10,
    )
    classification_colorbar.ax.tick_params(
        axis="x",
        labelsize=18,
    )

    # classification_figure.suptitle(
    #     f"广东省整月火灾阈值分类——{model_display_name}\n"
    #     f"{classification_description}",
    #     fontsize=22,
    #     fontweight="bold",
    # )

    classification_output_path = (
        figure_dir
        / f"fire_classification_month_{model_name}.png"
    )
    pdf_classification_output_path = (
        figure_dir
        / f"fire_classification_month_{model_name}.pdf"
    )

    classification_figure.savefig(
        classification_output_path,
        dpi=FIGURE_DPI,
        bbox_inches=None,
    )
    classification_figure.savefig(
        pdf_classification_output_path,
        dpi=FIGURE_DPI,
        bbox_inches=None,
    )
    plt.close(classification_figure)

    print("[图像]", classification_output_path)
    output_paths.append(classification_output_path)

    return output_paths

# =============================================================================
# 9. 主流程
# =============================================================================
def main() -> None:
    dates = date_range(START_DATE, END_DATE)
    if not dates:
        raise ValueError("日期范围为空，请检查 START_DATE 和 END_DATE。")

    DELIVERY_ROOT.mkdir(parents=True, exist_ok=True)

    print("[1/7] 读取行政区边界")
    province_gdf, district_gdf = load_administrative_boundaries()

    print("[2/7] 构建区县编号栅格和点位元数据")
    target_crs, transform, height, width, bounds = load_target_grid()

    if target_crs.to_epsg() != 4326:
        raise ValueError(
            "当前 Pangu 插值和经纬度表要求目标网格为 EPSG:4326，"
            f"实际为 {target_crs}。"
        )

    guangdong_mask = np.load(GUANGDONG_MASK_PATH).astype(bool)
    if guangdong_mask.shape != (height, width):
        raise ValueError(
            f"广东掩膜 shape={guangdong_mask.shape}，"
            f"目标网格={(height, width)}"
        )

    district_grid = build_district_grid(
        district_gdf,
        target_crs,
        transform,
        height,
        width,
        guangdong_mask,
    )
    point_metadata = build_point_metadata(
        district_grid,
        district_gdf,
        transform,
    )

    all_outputs: List[Path] = []

    if EXPORT_MODEL_TABLES:
        print("[3/7] 导出每个模型的简单 CSV")
        for model_name in MODEL_NAMES:
            print("   MODEL", model_name)
            all_outputs.extend(
                export_model_tables(
                    model_name,
                    dates,
                    (height, width),
                    point_metadata,
                    district_gdf,
                )
            )

    if PREPARE_WEATHER_CACHE:
        print("[4/7] 匹配并缓存 19 个气象变量")
        source_rows, source_cols = pangu_source_indices(
            transform,
            height,
            width,
        )
        prepare_weather_cache(
            dates,
            source_rows,
            source_cols,
            guangdong_mask,
        )

    if EXPORT_WEATHER_TABLES:
        print("[5/7] 导出各气象变量简单 CSV")
        all_outputs.extend(
            export_weather_tables(
                dates,
                point_metadata,
                district_gdf,
            )
        )

    if GENERATE_WEATHER_GIF:
        print("[6/7] 生成 8 变量气象 GIF")
        all_outputs.append(
            create_weather_gif(
                dates,
                guangdong_mask,
                province_gdf,
                bounds,
            )
        )

    if GENERATE_FIRE_FIGURES:
        print("[7/7] 生成每个模型的整月火灾概率图和分类图")
        for model_name in MODEL_NAMES:
            all_outputs.extend(
                create_fire_probability_figure(
                    model_name,
                    dates,
                    (height, width),
                    guangdong_mask,
                    point_metadata,
                    district_gdf,
                    province_gdf,
                    bounds,
                )
            )

    print("\n[DONE]")
    print("Delivery root:", DELIVERY_ROOT.resolve())
    print("District count:", len(district_gdf))
    print("Point count:", len(point_metadata.flat_indices))
    print("Generated delivery files:", len(all_outputs))


if __name__ == "__main__":
    main()