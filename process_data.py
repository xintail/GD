#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
历史训练数据预处理：将每日 ERA5 GRIB、植被、人口和 FIRMS
统一匹配到广东省 FIRMS 1 km 网格。

重要说明
--------
- 本脚本完全不读取气象 NPY。
- 每天直接读取一个 upper GRIB 和一个 surface GRIB。
- GRIB 只在内存中转换为数组，不保存全球 NPY。
- FIRMS 保留原始 confidence，不进行二值化。

每日输出
--------
1. feature_YYYYMMDD.npy
   shape = (72, H, W)

   通道顺序：
   0～3：
       [msl, 10u, 10v, 2t]

   4～68：
       [z, q, t, u, v] × 13 个气压层

   气压层顺序：
       [1000, 925, 850, 700, 600, 500, 400,
        300, 250, 200, 150, 100, 50]

   69：Fpar
   70：Lai
   71：Population

2. firms_conf_YYYYMMDD.npy
   shape = (H, W)
   保留 FIRMS confidence 原值。

数据清理
--------
Fpar：
    原始有效范围 0～100
    超出范围、NaN、Inf、NoData -> 0
    最后乘 0.01，输出范围 0～1

Lai：
    原始有效范围 0～100
    超出范围、NaN、Inf、NoData -> 0
    最后乘 0.1，输出范围 0～10

Population：
    有效范围 0～21171
    超出范围、NaN、Inf、NoData -> 0

依赖
----
conda install -c conda-forge \
    numpy scipy rasterio geopandas pyogrio shapely \
    xarray cfgrib eccodes
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
from scipy.ndimage import map_coordinates

try:
    import cfgrib
except ImportError as exc:
    raise ImportError(
        "缺少 cfgrib/eccodes，请执行：\n"
        "conda install -c conda-forge xarray cfgrib eccodes"
    ) from exc


# =============================================================================
# 0. 配置
# =============================================================================
START_DATE = "2025-12-01"
END_DATE = "2025-12-31"  # 包含当天

# 每日全球 ERA5 GRIB
UPPER_GRIB_DIR = Path("./data/raw/ERA5")
SURFACE_GRIB_DIR = Path("./data/raw/ERA5")

UPPER_GRIB_PATTERN = "input_upper_{date}.grib"
SURFACE_GRIB_PATTERN = "input_surface_{date}.grib"

# 广东 LAI/FPAR GeoTIFF；文件名中必须包含 YYYYMMDD
VEGETATION_DIR = Path("./data/raw/Fpar_Lai")
VEGETATION_GLOB = "*.tif"

# 广东人口 GeoTIFF
POPULATION_PATH = Path(
    "./data/raw/Population/GD_LandScan_Population_2024_500m.tif"
)

# 广东每日 FIRMS 1 km confidence GeoTIFF
FIRMS_DIR = Path("./data/raw/FIRMS")
FIRMS_PATTERN = "GD_FIRMS_conf_{date}_MODIS1km.tif"

# 广东边界
BOUNDARY_SHP = Path("./data/boundary/gd_boundary.shp")

# 输出
FEATURE_OUTPUT_DIR = Path("./data/processed/features_1km")
FIRMS_OUTPUT_DIR = Path("./data/processed/firms_conf_1km")
METADATA_PATH = Path("./data/processed/feature_channel_metadata.json")
GRID_METADATA_PATH = Path("./data/processed/target_grid_metadata.json")
MASK_OUTPUT_PATH = Path("./data/processed/guangdong_mask_1km.npy")

# 缺少某日文件时：
# "error"：立即报错
# "skip" ：跳过该日
MISSING_BEHAVIOR = "error"

# 已有结果是否覆盖
OVERWRITE = False

# 重采样方式
WEATHER_INTERPOLATION_ORDER = 1  # 1 = 双线性
VEGETATION_RESAMPLING = Resampling.bilinear
POPULATION_RESAMPLING = Resampling.bilinear
FIRMS_RESAMPLING = Resampling.nearest

# 原始有效范围和缩放系数
FPAR_RAW_MIN = 0.0
FPAR_RAW_MAX = 100.0
FPAR_SCALE = 0.01

LAI_RAW_MIN = 0.0
LAI_RAW_MAX = 100.0
LAI_SCALE = 0.1

POPULATION_MIN = 0.0
POPULATION_MAX = 21171.0

OUTSIDE_VALUE = 0.0

SURFACE_NAMES = ["msl", "10u", "10v", "2t"]
UPPER_NAMES = ["z", "q", "t", "u", "v"]
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50,]


# =============================================================================
# 1. 日期和文件匹配
# =============================================================================
def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d")


def date_range(start: str, end: str) -> Iterable[datetime]:
    current = parse_date(start)
    end_dt = parse_date(end)

    if end_dt < current:
        raise ValueError("END_DATE 不能早于 START_DATE。")

    while current <= end_dt:
        yield current
        current += timedelta(days=1)


def extract_yyyymmdd(path: Path) -> str:
    matches = re.findall(r"(?<!\d)(20\d{6})(?!\d)", path.name)
    if not matches:
        raise ValueError(
            "无法从植被文件名中提取 YYYYMMDD：" + path.name
        )
    return matches[-1]


def build_vegetation_index(
    directory: Path,
    pattern: str,
) -> Dict[str, Path]:
    files = sorted(directory.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"植被目录中没有找到文件：{directory} / {pattern}"
        )

    index: Dict[str, Path] = {}

    for path in files:
        index[extract_yyyymmdd(path)] = path

    return index


def nearest_vegetation_file(
    target_date: datetime,
    vegetation_index: Dict[str, Path],
) -> Tuple[str, Path]:
    candidates = []

    for date_text, path in vegetation_index.items():
        image_date = datetime.strptime(date_text, "%Y%m%d")
        distance = abs((image_date - target_date).days)

        # 距离相同时优先目标日之前的影像。
        future_flag = 1 if image_date > target_date else 0

        candidates.append(
            (distance, future_flag, image_date, date_text, path)
        )

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    _, _, _, date_text, path = candidates[0]
    return date_text, path


# =============================================================================
# 2. FIRMS 目标网格和广东掩膜
# =============================================================================
def find_reference_firms_file() -> Path:
    for day in date_range(START_DATE, END_DATE):
        date_text = day.strftime("%Y%m%d")
        path = FIRMS_DIR / FIRMS_PATTERN.format(date=date_text)
        
        if path.exists():
            return path

    raise FileNotFoundError(
        "日期范围内没有 FIRMS GeoTIFF，无法确定目标 1 km 网格。\n"
        f"目录：{FIRMS_DIR}\n"
        f"模板：{FIRMS_PATTERN}"
    )


def read_target_grid(reference_path: Path):
    with rasterio.open(reference_path) as src:
        if src.crs is None:
            raise ValueError(
                "FIRMS 参考文件没有 CRS：" + str(reference_path)
            )

        return (src.crs, src.transform, src.height, src.width, src.bounds,)


def build_guangdong_mask(
    boundary_path: Path,
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    if not boundary_path.exists():
        raise FileNotFoundError(
            "找不到广东边界：" + str(boundary_path)
        )

    boundary = gpd.read_file(boundary_path)

    if boundary.empty:
        raise ValueError("广东边界文件为空。")

    if boundary.crs is None:
        raise ValueError(
            "广东边界没有坐标系，请检查 gd_boundary.prj。"
        )

    boundary = boundary.to_crs(target_crs)
    boundary = boundary[
        boundary.geometry.notna()
        & ~boundary.geometry.is_empty
    ]

    geometries = list(boundary.geometry)

    if not geometries:
        raise ValueError("广东边界中没有有效几何。")

    return geometry_mask(
        geometries,
        out_shape=(target_height, target_width),
        transform=target_transform,
        invert=True,
        all_touched=False,
    )


# =============================================================================
# 3. 每日 GRIB 读取
# =============================================================================
def collect_grib_variables(
    grib_path: Path,
) -> Tuple[Dict[str, object], List[object]]:
    """
    一个 GRIB 文件只有一个时次。
    cfgrib 可能按变量类型拆成多个 Dataset，因此统一收集变量。

    返回：
        variables：变量名 -> DataArray
        datasets：用于处理完成后关闭文件
    """
    try:
        datasets = cfgrib.open_datasets(
            str(grib_path),
            backend_kwargs={"indexpath": ""},
        )
    except TypeError:
        # 兼容部分旧版本 cfgrib。
        datasets = cfgrib.open_datasets(str(grib_path))
    except Exception as exc:
        raise RuntimeError(
            "读取 GRIB 失败：" + str(grib_path)
        ) from exc

    variables: Dict[str, object] = {}

    for dataset in datasets:
        for name in dataset.data_vars:
            variables[name] = dataset[name]

    if not variables:
        for dataset in datasets:
            dataset.close()

        raise ValueError(
            "GRIB 中没有读取到变量：" + str(grib_path)
        )

    return variables, datasets


def close_datasets(datasets: Sequence[object]) -> None:
    for dataset in datasets:
        try:
            dataset.close()
        except Exception:
            pass


def find_variable(
    variables: Dict[str, object],
    aliases: Sequence[str],
    label: str,
):
    for alias in aliases:
        if alias in variables:
            return variables[alias]

    raise KeyError(
        "GRIB 中找不到 {}。候选名={}，实际变量={}".format(
            label,
            list(aliases),
            list(variables.keys()),
        )
    )


def find_dimension(
    data_array,
    candidates: Sequence[str],
    label: str,
) -> str:
    for name in candidates:
        if name in data_array.dims:
            return name

    raise ValueError(
        "{} 中找不到 {} 维度。现有维度={}".format(
            data_array.name,
            label,
            list(data_array.dims),
        )
    )


def extract_lat_lon(
    data_array,
) -> Tuple[np.ndarray, np.ndarray]:
    lat_dim = find_dimension(
        data_array,
        ["latitude", "lat"],
        "纬度",
    )
    lon_dim = find_dimension(
        data_array,
        ["longitude", "lon"],
        "经度",
    )

    latitude = np.asarray(
        data_array[lat_dim].values,
        dtype=np.float64,
    )
    longitude = np.asarray(
        data_array[lon_dim].values,
        dtype=np.float64,
    )

    if latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError("只支持规则一维经纬度网格。")

    return latitude, longitude


def upper_dataarray_to_numpy(
    data_array,
    variable_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    转为 shape=(13, H, W)，并返回该变量的纬度和经度。
    """
    data_array = data_array.squeeze(drop=True)

    level_dim = find_dimension(
        data_array,
        ["isobaricInhPa", "pressure_level", "level"],
        "气压层",
    )
    lat_dim = find_dimension(
        data_array,
        ["latitude", "lat"],
        "纬度",
    )
    lon_dim = find_dimension(
        data_array,
        ["longitude", "lon"],
        "经度",
    )

    levels = np.asarray(
        data_array[level_dim].values,
        dtype=np.float64,
    )

    if levels.max() > 2000:
        levels_hpa = levels / 100.0
    else:
        levels_hpa = levels

    indexes: List[int] = []

    for wanted in PRESSURE_LEVELS:
        matched = np.where(
            np.isclose(levels_hpa, wanted, atol=1e-5)
        )[0]

        if matched.size == 0:
            raise ValueError(
                "{} 缺少 {} hPa。现有气压层={}".format(
                    variable_name,
                    wanted,
                    levels_hpa.tolist(),
                )
            )

        indexes.append(int(matched[0]))

    data_array = data_array.isel(
        {level_dim: indexes}
    ).transpose(level_dim, lat_dim, lon_dim)

    array = np.asarray(
        data_array.values,
        dtype=np.float32,
    )

    latitude = np.asarray(
        data_array[lat_dim].values,
        dtype=np.float64,
    )
    longitude = np.asarray(
        data_array[lon_dim].values,
        dtype=np.float64,
    )

    if array.shape[0] != 13:
        raise ValueError(
            "{} 气压层数量错误：{}".format(
                variable_name,
                array.shape,
            )
        )

    return array, latitude, longitude


def surface_dataarray_to_numpy(
    data_array,
    variable_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    转为 shape=(H, W)，并返回该变量的纬度和经度。
    """
    data_array = data_array.squeeze(drop=True)

    lat_dim = find_dimension(
        data_array,
        ["latitude", "lat"],
        "纬度",
    )
    lon_dim = find_dimension(
        data_array,
        ["longitude", "lon"],
        "经度",
    )

    data_array = data_array.transpose(lat_dim, lon_dim)

    array = np.asarray(
        data_array.values,
        dtype=np.float32,
    )

    latitude = np.asarray(
        data_array[lat_dim].values,
        dtype=np.float64,
    )
    longitude = np.asarray(
        data_array[lon_dim].values,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            "{} 转换后不是二维数组：{}".format(
                variable_name,
                array.shape,
            )
        )

    return array, latitude, longitude


def grids_equal(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> bool:
    return (
        lat1.shape == lat2.shape
        and lon1.shape == lon2.shape
        and np.allclose(lat1, lat2, atol=1e-8)
        and np.allclose(lon1, lon2, atol=1e-8)
    )


def load_daily_weather_grib(
    upper_grib_path: Path,
    surface_grib_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    直接读取每日 GRIB。

    返回：
        upper      shape=(5,13,H,W)
        surface    shape=(4,H,W)
        latitude   shape=(H,)
        longitude  shape=(W,)
    """
    upper_variables, upper_datasets = collect_grib_variables(
        upper_grib_path
    )
    surface_variables, surface_datasets = collect_grib_variables(
        surface_grib_path
    )

    try:
        upper_arrays: List[np.ndarray] = []
        surface_arrays: List[np.ndarray] = []

        reference_lat: Optional[np.ndarray] = None
        reference_lon: Optional[np.ndarray] = None

        upper_aliases = {
            "z": ["z"],
            "q": ["q"],
            "t": ["t"],
            "u": ["u"],
            "v": ["v"],
        }

        for variable_name in UPPER_NAMES:
            data_array = find_variable(
                upper_variables,
                upper_aliases[variable_name],
                variable_name,
            )

            array, latitude, longitude = upper_dataarray_to_numpy(
                data_array,
                variable_name,
            )

            if reference_lat is None:
                reference_lat = latitude
                reference_lon = longitude
            elif not grids_equal(
                reference_lat,
                reference_lon,
                latitude,
                longitude,
            ):
                raise ValueError(
                    f"upper 变量 {variable_name} 的经纬度网格不一致。"
                )

            upper_arrays.append(array)

        surface_aliases = {
            "msl": ["msl", "prmsl"],
            "10u": ["u10", "10u"],
            "10v": ["v10", "10v"],
            "2t": ["t2m", "2t"],
        }

        for variable_name in SURFACE_NAMES:
            data_array = find_variable(
                surface_variables,
                surface_aliases[variable_name],
                variable_name,
            )

            array, latitude, longitude = surface_dataarray_to_numpy(
                data_array,
                variable_name,
            )

            if reference_lat is None:
                reference_lat = latitude
                reference_lon = longitude
            elif not grids_equal(
                reference_lat,
                reference_lon,
                latitude,
                longitude,
            ):
                raise ValueError(
                    f"surface 变量 {variable_name} 的经纬度网格不一致。"
                )

            surface_arrays.append(array)

        if reference_lat is None or reference_lon is None:
            raise RuntimeError("未能读取气象经纬度。")

        upper = np.stack(
            upper_arrays,
            axis=0,
        ).astype(np.float32, copy=False)

        surface = np.stack(
            surface_arrays,
            axis=0,
        ).astype(np.float32, copy=False)

        print(
            "  upper shape={}, surface shape={}".format(
                upper.shape,
                surface.shape,
            )
        )
        print(
            "  latitude={} -> {}, longitude={} -> {}".format(
                reference_lat[0],
                reference_lat[-1],
                reference_lon[0],
                reference_lon[-1],
            )
        )

        return (
            upper,
            surface,
            reference_lat,
            reference_lon,
        )

    finally:
        close_datasets(upper_datasets)
        close_datasets(surface_datasets)


# =============================================================================
# 4. 气象匹配到广东 1 km
# =============================================================================
def target_lon_lat(
    transform,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(
        (height, width),
        dtype=np.float64,
    )

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


def build_weather_source_indices(
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    target_transform,
    target_height: int,
    target_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 GRIB 自带经纬度坐标计算目标像元在源网格中的浮点索引。
    不要求经度一定是 0～360，也不要求纬度一定升序。
    """
    target_lon, target_lat = target_lon_lat(
        target_transform,
        target_height,
        target_width,
    )

    if source_latitude.size < 2 or source_longitude.size < 2:
        raise ValueError("气象经纬度坐标长度不足。")

    lat_step = float(source_latitude[1] - source_latitude[0])
    lon_step = float(source_longitude[1] - source_longitude[0])

    if np.isclose(lat_step, 0.0) or np.isclose(lon_step, 0.0):
        raise ValueError("气象经纬度分辨率为 0。")

    # 根据源经度范围调整广东目标经度。
    if source_longitude.min() >= 0:
        target_lon = np.mod(target_lon, 360.0)
    else:
        target_lon = (
            (target_lon + 180.0) % 360.0
        ) - 180.0

    source_rows = (
        target_lat - float(source_latitude[0])
    ) / lat_step

    source_cols = (
        target_lon - float(source_longitude[0])
    ) / lon_step

    return source_rows, source_cols


def resample_weather_channel(
    source: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)

    result = map_coordinates(
        source,
        [source_rows, source_cols],
        order=WEATHER_INTERPOLATION_ORDER,
        mode="nearest",
        prefilter=False,
    )

    return result.astype(np.float32, copy=False)


def match_weather_to_target(
    upper: np.ndarray,
    surface: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    target_transform,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    source_rows, source_cols = build_weather_source_indices(
        source_latitude,
        source_longitude,
        target_transform,
        target_height,
        target_width,
    )

    channels: List[np.ndarray] = []

    # 4 个地表变量
    for channel_index in range(4):
        channels.append(
            resample_weather_channel(
                surface[channel_index],
                source_rows,
                source_cols,
            )
        )

    # 5 个高空变量 × 13 个气压层
    for variable_index in range(5):
        for level_index in range(13):
            channels.append(
                resample_weather_channel(
                    upper[variable_index, level_index],
                    source_rows,
                    source_cols,
                )
            )

    weather = np.stack(
        channels,
        axis=0,
    ).astype(np.float32, copy=False)

    if weather.shape != (
        69,
        target_height,
        target_width,
    ):
        raise ValueError(
            "气象匹配结果 shape 错误：{}".format(
                weather.shape
            )
        )

    return weather


# =============================================================================
# 5. GeoTIFF 清理和重采样
# =============================================================================
def same_grid(
    src,
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
) -> bool:
    return (
        src.crs == target_crs
        and src.transform.almost_equals(target_transform)
        and src.height == target_height
        and src.width == target_width
    )


def clean_source_array(
    source: np.ndarray,
    nodata: Optional[float],
    valid_min: Optional[float],
    valid_max: Optional[float],
) -> np.ndarray:
    source = np.asarray(
        source,
        dtype=np.float32,
    ).copy()

    invalid = ~np.isfinite(source)

    if nodata is not None and np.isfinite(nodata):
        invalid |= np.isclose(source, nodata)

    if valid_min is not None:
        invalid |= source < valid_min

    if valid_max is not None:
        invalid |= source > valid_max

    source[invalid] = 0.0
    return source


def reproject_clean_band(
    src,
    band_index: int,
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
    resampling: Resampling,
    valid_min: Optional[float],
    valid_max: Optional[float],
) -> np.ndarray:
    source = src.read(
        band_index
    ).astype(np.float32)

    source = clean_source_array(
        source,
        src.nodata,
        valid_min,
        valid_max,
    )

    if same_grid(
        src,
        target_crs,
        target_transform,
        target_height,
        target_width,
    ):
        destination = source
    else:
        destination = np.zeros(
            (target_height, target_width),
            dtype=np.float32,
        )

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=None,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=0.0,
            resampling=resampling,
        )

    destination = np.nan_to_num(
        destination,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

    return destination


def load_vegetation(
    path: Path,
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """
    假定植被 GeoTIFF：
        band 1 = Fpar
        band 2 = Lai
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                "植被 GeoTIFF 没有 CRS：" + str(path)
            )

        if src.count < 2:
            raise ValueError(
                "植被文件至少需要两个波段 [Fpar, Lai]："
                + str(path)
            )

        fpar = reproject_clean_band(
            src=src,
            band_index=1,
            target_crs=target_crs,
            target_transform=target_transform,
            target_height=target_height,
            target_width=target_width,
            resampling=VEGETATION_RESAMPLING,
            valid_min=FPAR_RAW_MIN,
            valid_max=FPAR_RAW_MAX,
        )

        lai = reproject_clean_band(
            src=src,
            band_index=2,
            target_crs=target_crs,
            target_transform=target_transform,
            target_height=target_height,
            target_width=target_width,
            resampling=VEGETATION_RESAMPLING,
            valid_min=LAI_RAW_MIN,
            valid_max=LAI_RAW_MAX,
        )

    # 重采样后再次严格检查范围。
    fpar[
        (~np.isfinite(fpar))
        | (fpar < FPAR_RAW_MIN)
        | (fpar > FPAR_RAW_MAX)
    ] = 0.0

    lai[
        (~np.isfinite(lai))
        | (lai < LAI_RAW_MIN)
        | (lai > LAI_RAW_MAX)
    ] = 0.0

    # 应用比例因子。
    fpar *= FPAR_SCALE
    lai *= LAI_SCALE

    return np.stack(
        [fpar, lai],
        axis=0,
    ).astype(np.float32, copy=False)


def load_population(
    path: Path,
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                "人口 GeoTIFF 没有 CRS：" + str(path)
            )

        population = reproject_clean_band(
            src=src,
            band_index=1,
            target_crs=target_crs,
            target_transform=target_transform,
            target_height=target_height,
            target_width=target_width,
            resampling=POPULATION_RESAMPLING,
            valid_min=POPULATION_MIN,
            valid_max=POPULATION_MAX,
        )

    population[
        (~np.isfinite(population))
        | (population < POPULATION_MIN)
        | (population > POPULATION_MAX)
    ] = 0.0

    return population[
        np.newaxis, ...
    ].astype(np.float32, copy=False)


def load_firms_confidence(
    path: Path,
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                "FIRMS GeoTIFF 没有 CRS：" + str(path)
            )

        firms = reproject_clean_band(
            src=src,
            band_index=1,
            target_crs=target_crs,
            target_transform=target_transform,
            target_height=target_height,
            target_width=target_width,
            resampling=FIRMS_RESAMPLING,
            valid_min=None,
            valid_max=None,
        )

    # 只清理非法数值，不进行阈值化或二值化。
    firms = np.nan_to_num(
        firms,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

    return firms


# =============================================================================
# 6. 元数据
# =============================================================================
def build_channel_metadata() -> List[Dict[str, object]]:
    metadata: List[Dict[str, object]] = []
    channel = 0

    for variable in SURFACE_NAMES:
        metadata.append(
            {
                "channel": channel,
                "source": "weather_surface",
                "variable": variable,
                "pressure_level_hpa": None,
            }
        )
        channel += 1

    for variable in UPPER_NAMES:
        for level in PRESSURE_LEVELS:
            metadata.append(
                {
                    "channel": channel,
                    "source": "weather_upper",
                    "variable": variable,
                    "pressure_level_hpa": level,
                }
            )
            channel += 1

    metadata.append(
        {
            "channel": channel,
            "source": "vegetation",
            "variable": "Fpar",
            "raw_valid_range": [
                FPAR_RAW_MIN,
                FPAR_RAW_MAX,
            ],
            "scale": FPAR_SCALE,
            "output_valid_range": [0.0, 1.0],
        }
    )
    channel += 1

    metadata.append(
        {
            "channel": channel,
            "source": "vegetation",
            "variable": "Lai",
            "raw_valid_range": [
                LAI_RAW_MIN,
                LAI_RAW_MAX,
            ],
            "scale": LAI_SCALE,
            "output_valid_range": [0.0, 10.0],
        }
    )
    channel += 1

    metadata.append(
        {
            "channel": channel,
            "source": "population",
            "variable": "population",
            "valid_range": [
                POPULATION_MIN,
                POPULATION_MAX,
            ],
        }
    )

    if len(metadata) != 72:
        raise RuntimeError(
            f"通道数量错误：{len(metadata)}，应为 72。"
        )

    return metadata


# =============================================================================
# 7. 主流程
# =============================================================================
def main() -> None:
    if MISSING_BEHAVIOR not in {"error", "skip"}:
        raise ValueError(
            "MISSING_BEHAVIOR 必须是 'error' 或 'skip'。"
        )

    FEATURE_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    FIRMS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    METADATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not POPULATION_PATH.exists():
        raise FileNotFoundError(
            "找不到人口文件：" + str(POPULATION_PATH)
        )

    vegetation_index = build_vegetation_index(
        VEGETATION_DIR,
        VEGETATION_GLOB,
    )

    reference_firms = find_reference_firms_file()

    (
        target_crs,
        target_transform,
        target_height,
        target_width,
        target_bounds,
    ) = read_target_grid(reference_firms)

    print("[INFO] reference FIRMS:", reference_firms)
    print("[INFO] target CRS:", target_crs)
    print("[INFO] target shape:", (target_height, target_width))
    print("[INFO] target bounds:", target_bounds)

    if target_crs.to_epsg() != 4326:
        raise ValueError(
            "当前气象匹配要求 FIRMS 网格为 EPSG:4326，"
            f"当前 CRS={target_crs}"
        )

    guangdong_mask = build_guangdong_mask(
        BOUNDARY_SHP,
        target_crs,
        target_transform,
        target_height,
        target_width,
    )

    np.save(
        MASK_OUTPUT_PATH,
        guangdong_mask.astype(np.uint8),
    )

    # 人口为静态变量，只匹配一次。
    print("[INFO] matching population:", POPULATION_PATH)

    population = load_population(
        POPULATION_PATH,
        target_crs,
        target_transform,
        target_height,
        target_width,
    )

    population[:, ~guangdong_mask] = OUTSIDE_VALUE

    channel_metadata = build_channel_metadata()

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "feature_shape": [
                    72,
                    target_height,
                    target_width,
                ],
                "channel_order": channel_metadata,
                "surface_variable_order": SURFACE_NAMES,
                "upper_variable_order": UPPER_NAMES,
                "pressure_level_order_hpa": PRESSURE_LEVELS,
                "weather_input": {
                    "format": "daily GRIB only",
                    "global_npy_used": False,
                    "global_npy_saved": False,
                },
                "firms_binarized": False,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        GRID_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "reference_firms_file": str(reference_firms),
                "crs": target_crs.to_string(),
                "height": target_height,
                "width": target_width,
                "bounds": {
                    "left": target_bounds.left,
                    "bottom": target_bounds.bottom,
                    "right": target_bounds.right,
                    "top": target_bounds.top,
                },
                "transform": list(target_transform)[:6],
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    success_count = 0
    skip_count = 0

    for day in date_range(START_DATE, END_DATE):
        date_text = day.strftime("%Y%m%d")

        feature_out = (
            FEATURE_OUTPUT_DIR
            / f"feature_{date_text}.npy"
        )
        firms_out = (
            FIRMS_OUTPUT_DIR
            / f"firms_conf_{date_text}.npy"
        )

        if (
            not OVERWRITE
            and feature_out.exists()
            and firms_out.exists()
        ):
            print("[SKIP] outputs already exist:", date_text)
            skip_count += 1
            continue

        upper_grib_path = (
            UPPER_GRIB_DIR
            / UPPER_GRIB_PATTERN.format(date=date_text)
        )
        surface_grib_path = (
            SURFACE_GRIB_DIR
            / SURFACE_GRIB_PATTERN.format(date=date_text)
        )
        firms_path = (
            FIRMS_DIR
            / FIRMS_PATTERN.format(date=date_text)
        )

        required_paths = [
            upper_grib_path,
            surface_grib_path,
            firms_path,
        ]

        missing_paths = [
            path
            for path in required_paths
            if not path.exists()
        ]

        if missing_paths:
            message = (
                f"{date_text} 缺少文件：\n"
                + "\n".join(
                    "  " + str(path)
                    for path in missing_paths
                )
            )

            if MISSING_BEHAVIOR == "skip":
                print("[SKIP]", message)
                skip_count += 1
                continue

            raise FileNotFoundError(message)

        vegetation_date, vegetation_path = (
            nearest_vegetation_file(
                day,
                vegetation_index,
            )
        )

        print("\n[DAY]", date_text)
        print("  upper GRIB:", upper_grib_path)
        print("  surface GRIB:", surface_grib_path)
        print(
            "  vegetation: {} (matched {})".format(
                vegetation_path,
                vegetation_date,
            )
        )
        print("  FIRMS:", firms_path)

        (
            upper,
            surface,
            source_latitude,
            source_longitude,
        ) = load_daily_weather_grib(
            upper_grib_path,
            surface_grib_path,
        )

        weather = match_weather_to_target(
            upper=upper,
            surface=surface,
            source_latitude=source_latitude,
            source_longitude=source_longitude,
            target_transform=target_transform,
            target_height=target_height,
            target_width=target_width,
        )

        weather[:, ~guangdong_mask] = OUTSIDE_VALUE

        vegetation = load_vegetation(
            vegetation_path,
            target_crs,
            target_transform,
            target_height,
            target_width,
        )

        vegetation[:, ~guangdong_mask] = OUTSIDE_VALUE

        features = np.concatenate(
            [
                weather,       # 69
                vegetation,    # 2
                population,    # 1
            ],
            axis=0,
        ).astype(np.float32, copy=False)

        expected_shape = (
            72,
            target_height,
            target_width,
        )

        if features.shape != expected_shape:
            raise ValueError(
                "{} 特征 shape={}，期望 {}".format(
                    date_text,
                    features.shape,
                    expected_shape,
                )
            )

        firms_conf = load_firms_confidence(
            firms_path,
            target_crs,
            target_transform,
            target_height,
            target_width,
        )

        firms_conf[~guangdong_mask] = OUTSIDE_VALUE

        np.save(
            feature_out,
            np.ascontiguousarray(features),
        )
        np.save(
            firms_out,
            np.ascontiguousarray(firms_conf),
        )

        print(
            "[OK] feature: {} shape={} dtype={}".format(
                feature_out,
                features.shape,
                features.dtype,
            )
        )
        print(
            "[OK] FIRMS: {} shape={} min={} max={}".format(
                firms_out,
                firms_conf.shape,
                float(firms_conf.min()),
                float(firms_conf.max()),
            )
        )
        print(
            "[CHECK] Fpar: {} ～ {}".format(
                float(features[69].min()),
                float(features[69].max()),
            )
        )
        print(
            "[CHECK] Lai: {} ～ {}".format(
                float(features[70].min()),
                float(features[70].max()),
            )
        )
        print(
            "[CHECK] Population: {} ～ {}".format(
                float(features[71].min()),
                float(features[71].max()),
            )
        )

        success_count += 1

        # 明确释放当天全球气象数组。
        del upper
        del surface
        del weather
        del vegetation
        del features
        del firms_conf

    print("\n[DONE]")
    print("  success:", success_count)
    print("  skipped:", skip_count)
    print("  feature dir:", FEATURE_OUTPUT_DIR.resolve())
    print("  FIRMS dir:", FIRMS_OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()