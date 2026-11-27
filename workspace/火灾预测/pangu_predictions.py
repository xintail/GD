#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用全球 Pangu-Weather NPY 直接进行广东省火灾预测。

处理流程
--------
全球 Pangu upper/surface NPY
        ↓
匹配到训练阶段的广东 FIRMS 1 km 网格
        ↓
加入最近日期的 Fpar/Lai 和静态 Population
        ↓
按照 checkpoint 保存的通道顺序构造模型输入
        ↓
使用各模型保存的 Pipeline 和最佳 F1 阈值进行预测
        ↓
输出各模型概率、二值结果和多模型集成结果

重要说明
--------
1. 不需要预先生成 feature_YYYYMMDD.npy。
2. 不会保存完整 72 通道特征，减少磁盘占用。
3. sklearn Pipeline 中已经保存了训练阶段的：
       SimpleImputer
       StandardScaler
       分类模型
   因此预测时不能重新拟合归一化器。
4. 模型只对 guangdong_mask_1km.npy 内的广东像元进行预测。
5. 预测时不需要 FIRMS 标签。

Pangu 输入要求
--------------
surface：
    shape = (4, 721, 1440)
    顺序 = [msl, 10u, 10v, 2t]

upper：
    shape = (5, 13, 721, 1440)
    变量顺序 = [z, q, t, u, v]
    气压层顺序 =
        [1000, 925, 850, 700, 600, 500, 400,
         300, 250, 200, 150, 100, 50]

默认文件名
----------
./Pangu-Weather-main/Pangu_npy/input_upper_YYYYMMDD.npy
./Pangu-Weather-main/Pangu_npy/input_surface_YYYYMMDD.npy

输出
----
predictions_from_pangu/<model_name>/
    fire_probability_YYYYMMDD.npy
    fire_binary_YYYYMMDD.npy
    prediction_summary_YYYYMMDD.json

predictions_from_pangu/ensemble/
    mean_probability_YYYYMMDD.npy
    majority_vote_YYYYMMDD.npy
    prediction_summary_YYYYMMDD.json

依赖
----
conda install -c conda-forge \
    numpy scipy rasterio geopandas shapely joblib scikit-learn
"""

from __future__ import annotations
import sys
sys.path.append('./')
import gc
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import rasterio
from rasterio import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.ndimage import map_coordinates

from model.calibrated_probability_model import CalibratedProbabilityModel

# =============================================================================
# 0. 配置
# =============================================================================

# 需要进行火灾预测的日期范围
START_DATE = "2026-12-01"
END_DATE = "2026-12-31"  # 包含当天

# ---------------------------------------------------------------------------
# 全球 Pangu-Weather NPY
# ---------------------------------------------------------------------------
PANGU_NPY_DIR = Path("./Pangu-Weather-main/Pangu_npy")

UPPER_PATTERN = "input_upper_{date}.npy"
SURFACE_PATTERN = "input_surface_{date}.npy"

# 如果你的文件名包含小时，例如 input_upper_2025100100.npy，
# 可改为：
# UPPER_PATTERN = "input_upper_{date}00.npy"
# SURFACE_PATTERN = "input_surface_{date}00.npy"

# ---------------------------------------------------------------------------
# 植被和人口必须与训练阶段使用相同的数据处理方式
# ---------------------------------------------------------------------------
VEGETATION_DIR = Path("./data/raw/Fpar_Lai")
VEGETATION_GLOB = "*.tif"

POPULATION_PATH = Path("./data/raw/Population/GD_LandScan_Population_2024_500m.tif")

# ---------------------------------------------------------------------------
# 训练阶段产生的空间信息
# ---------------------------------------------------------------------------
GRID_METADATA_PATH = Path(
    "./data/processed/target_grid_metadata.json"
)
MASK_PATH = Path(
    "./data/processed/guangdong_mask_1km.npy"
)

# ---------------------------------------------------------------------------
# 训练好的模型
# ---------------------------------------------------------------------------
# CHECKPOINT_ROOT = Path("./checkpoint")
# PREDICTION_ROOT = Path("./workspace/火灾预测/pangu_predictions")
CHECKPOINT_ROOT = Path("./checkpoint_calibration")
PREDICTION_ROOT = Path("./workspace/火灾预测/pangu_predictions_calibration")

MODEL_NAMES = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
]

# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

OVERWRITE = False
PREDICT_BATCH_SIZE = 100000

# 是否额外保存当前日期匹配后的 22 通道模型输入。
# 通常不需要，保持 False 可减少磁盘占用。
SAVE_SELECTED_FEATURES = False
SELECTED_FEATURE_OUTPUT_DIR = Path(
    "./data/processed/pangu_selected_features_1km"
)

# ---------------------------------------------------------------------------
# 数据清理与重采样，必须和训练阶段一致
# ---------------------------------------------------------------------------
WEATHER_INTERPOLATION_ORDER = 1
VEGETATION_RESAMPLING = Resampling.bilinear
POPULATION_RESAMPLING = Resampling.bilinear

FPAR_RAW_MIN = 0.0
FPAR_RAW_MAX = 100.0
FPAR_SCALE = 0.01

LAI_RAW_MIN = 0.0
LAI_RAW_MAX = 100.0
LAI_SCALE = 0.1

POPULATION_MIN = 0.0
POPULATION_MAX = 21171.0

OUTSIDE_VALUE = 0.0

SURFACE_VARIABLE_INDEX = {
    "msl": 0,
    "10u": 1,
    "u10": 1,
    "10v": 2,
    "v10": 2,
    "2t": 3,
    "t2m": 3,
}

UPPER_VARIABLE_INDEX = {
    "z": 0,
    "q": 1,
    "t": 2,
    "u": 3,
    "v": 4,
}

PANGU_PRESSURE_LEVELS = [
    1000, 925, 850, 700, 600, 500, 400,
    300, 250, 200, 150, 100, 50,
]

EXPECTED_SURFACE_SHAPE = (4, 721, 1440)
EXPECTED_UPPER_SHAPE = (5, 13, 721, 1440)


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


def extract_yyyymmdd(path: Path) -> str:
    matches = re.findall(
        r"(?<!\d)(20\d{6})(?!\d)",
        path.name,
    )

    if not matches:
        raise ValueError(
            f"无法从植被文件名提取 YYYYMMDD：{path.name}"
        )

    return matches[-1]


# =============================================================================
# 2. 目标网格
# =============================================================================
def load_target_grid():
    if not GRID_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到目标网格元数据：{GRID_METADATA_PATH}"
        )

    metadata = load_json(GRID_METADATA_PATH)

    crs = rasterio.crs.CRS.from_string(
        metadata["crs"]
    )

    transform_values = metadata["transform"]

    if len(transform_values) != 6:
        raise ValueError(
            "target_grid_metadata.json 中 transform 应有 6 个数值。"
        )

    transform = Affine(*map(float, transform_values))
    height = int(metadata["height"])
    width = int(metadata["width"])

    return crs, transform, height, width


def load_mask(
    target_height: int,
    target_width: int,
) -> np.ndarray:
    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"找不到广东掩膜：{MASK_PATH}"
        )

    mask = np.load(MASK_PATH).astype(bool)

    if mask.shape != (target_height, target_width):
        raise ValueError(
            f"广东掩膜 shape={mask.shape}，"
            f"目标网格为 {(target_height, target_width)}"
        )

    return mask


def target_lon_lat(
    transform: Affine,
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


def build_pangu_source_indices(
    transform: Affine,
    target_height: int,
    target_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pangu 全球网格：
        latitude  = 90, 89.75, ..., -90
        longitude = 0, 0.25, ..., 359.75
    """
    longitude, latitude = target_lon_lat(
        transform,
        target_height,
        target_width,
    )

    longitude = np.mod(longitude, 360.0)

    source_rows = (90.0 - latitude) / 0.25
    source_cols = longitude / 0.25

    return source_rows, source_cols


# =============================================================================
# 3. Pangu 气象匹配
# =============================================================================
def load_pangu_weather(
    date_text: str,
) -> Tuple[np.ndarray, np.ndarray]:
    upper_path = (
        PANGU_NPY_DIR
        / UPPER_PATTERN.format(date=date_text)
    )
    surface_path = (
        PANGU_NPY_DIR
        / SURFACE_PATTERN.format(date=date_text)
    )

    if not upper_path.exists():
        raise FileNotFoundError(
            f"{date_text} 缺少 upper NPY：{upper_path}"
        )

    if not surface_path.exists():
        raise FileNotFoundError(
            f"{date_text} 缺少 surface NPY：{surface_path}"
        )

    upper = np.load(
        upper_path,
        mmap_mode="r",
    )
    surface = np.load(
        surface_path,
        mmap_mode="r",
    )

    if upper.shape != EXPECTED_UPPER_SHAPE:
        raise ValueError(
            f"{upper_path} shape={upper.shape}，"
            f"期望 {EXPECTED_UPPER_SHAPE}"
        )

    if surface.shape != EXPECTED_SURFACE_SHAPE:
        raise ValueError(
            f"{surface_path} shape={surface.shape}，"
            f"期望 {EXPECTED_SURFACE_SHAPE}"
        )

    print("  upper:", upper_path)
    print("  surface:", surface_path)

    return upper, surface


def resample_weather_channel(
    source: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
) -> np.ndarray:
    source = np.asarray(
        source,
        dtype=np.float32,
    )

    if source.shape != (721, 1440):
        raise ValueError(
            f"Pangu 单通道 shape={source.shape}，"
            "期望 (721, 1440)"
        )

    result = map_coordinates(
        source,
        [source_rows, source_cols],
        order=WEATHER_INTERPOLATION_ORDER,
        mode="nearest",
        prefilter=False,
    )

    return result.astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# 4. 植被和人口匹配
# =============================================================================
def build_vegetation_index() -> Dict[str, Path]:
    files = sorted(
        VEGETATION_DIR.glob(VEGETATION_GLOB)
    )

    if not files:
        raise FileNotFoundError(
            f"植被目录中没有文件："
            f"{VEGETATION_DIR} / {VEGETATION_GLOB}"
        )

    index: Dict[str, Path] = {}

    for path in files:
        index[extract_yyyymmdd(path)] = path

    return index


def nearest_vegetation_file(
    date_text: str,
    vegetation_index: Dict[str, Path],
) -> Tuple[str, Path]:
    target_date = datetime.strptime(
        date_text,
        "%Y%m%d",
    )

    candidates = []

    for image_date_text, path in vegetation_index.items():
        image_date = datetime.strptime(
            image_date_text,
            "%Y%m%d",
        )

        distance = abs(
            (image_date - target_date).days
        )

        # 距离相同时优先使用目标日之前的影像。
        future_flag = 1 if image_date > target_date else 0

        candidates.append(
            (
                distance,
                future_flag,
                image_date,
                image_date_text,
                path,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    _, _, _, matched_date, matched_path = candidates[0]
    return matched_date, matched_path


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
        invalid |= np.isclose(
            source,
            nodata,
        )

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
) -> Dict[str, np.ndarray]:
    """
    假定：
        band 1 = Fpar
        band 2 = Lai
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(
                f"植被 GeoTIFF 没有 CRS：{path}"
            )

        if src.count < 2:
            raise ValueError(
                f"植被文件至少需要两个波段 [Fpar, Lai]：{path}"
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

    fpar *= FPAR_SCALE
    lai *= LAI_SCALE

    return {
        "Fpar": fpar.astype(np.float32, copy=False),
        "Lai": lai.astype(np.float32, copy=False),
    }


def load_population(
    target_crs,
    target_transform,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    if not POPULATION_PATH.exists():
        raise FileNotFoundError(
            f"找不到人口文件：{POPULATION_PATH}"
        )

    with rasterio.open(POPULATION_PATH) as src:
        if src.crs is None:
            raise ValueError(
                f"人口 GeoTIFF 没有 CRS：{POPULATION_PATH}"
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

    return population.astype(
        np.float32,
        copy=False,
    )


# =============================================================================
# 5. 模型与通道配置
# =============================================================================
def load_models_and_configuration():
    models = {}
    thresholds = {}
    selected_metadata_by_model = {}

    for model_name in MODEL_NAMES:
        model_dir = CHECKPOINT_ROOT / model_name

        model_path = model_dir / "model.joblib"
        threshold_path = model_dir / "threshold.json"
        channels_path = model_dir / "selected_channels.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到模型：{model_path}"
            )

        if not threshold_path.exists():
            raise FileNotFoundError(
                f"找不到阈值：{threshold_path}"
            )

        if not channels_path.exists():
            raise FileNotFoundError(
                f"找不到通道配置：{channels_path}"
            )

        models[model_name] = joblib.load(
            model_path
        )

        thresholds[model_name] = float(
            load_json(threshold_path)["threshold"]
        )

        channel_payload = load_json(
            channels_path
        )

        selected_metadata_by_model[model_name] = (
            channel_payload["selected_channel_metadata"]
        )

        (PREDICTION_ROOT / model_name).mkdir(
            parents=True,
            exist_ok=True,
        )

    reference_metadata = selected_metadata_by_model[
        MODEL_NAMES[0]
    ]

    for model_name in MODEL_NAMES[1:]:
        if (
            selected_metadata_by_model[model_name]
            != reference_metadata
        ):
            raise ValueError(
                f"{model_name} 的选中特征与其他模型不一致。"
            )

    return (
        models,
        thresholds,
        reference_metadata,
    )


# =============================================================================
# 6. 构造模型输入
# =============================================================================
def resolve_upper_level_index(
    level_hpa: int,
) -> int:
    if level_hpa not in PANGU_PRESSURE_LEVELS:
        raise ValueError(
            f"checkpoint 请求 {level_hpa} hPa，"
            f"但 Pangu 输入层为 {PANGU_PRESSURE_LEVELS}"
        )

    return PANGU_PRESSURE_LEVELS.index(
        level_hpa
    )


def build_selected_feature_matrix(
    selected_metadata: Sequence[Dict[str, object]],
    upper: np.ndarray,
    surface: np.ndarray,
    vegetation: Dict[str, np.ndarray],
    population: np.ndarray,
    mask: np.ndarray,
    source_rows: np.ndarray,
    source_cols: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    直接构造模型需要的 22 个通道，不创建完整 72 通道数组。

    返回：
        x：广东掩膜内像元，shape=(N_pixels, N_features)
        selected_feature_cube：
            仅在 SAVE_SELECTED_FEATURES=True 时返回
            shape=(N_features, H, W)
    """
    pixel_count = int(mask.sum())
    feature_count = len(selected_metadata)

    x = np.empty(
        (pixel_count, feature_count),
        dtype=np.float32,
    )

    if SAVE_SELECTED_FEATURES:
        feature_cube = np.zeros(
            (
                feature_count,
                mask.shape[0],
                mask.shape[1],
            ),
            dtype=np.float32,
        )
    else:
        feature_cube = None

    for feature_index, item in enumerate(
        selected_metadata
    ):
        source = item["source"]
        variable = str(item["variable"])

        if source == "weather_surface":
            if variable not in SURFACE_VARIABLE_INDEX:
                raise KeyError(
                    f"不支持的 surface 变量：{variable}"
                )

            source_channel = surface[
                SURFACE_VARIABLE_INDEX[variable]
            ]

            feature_map = resample_weather_channel(
                source_channel,
                source_rows,
                source_cols,
            )

        elif source == "weather_upper":
            if variable not in UPPER_VARIABLE_INDEX:
                raise KeyError(
                    f"不支持的 upper 变量：{variable}"
                )

            level_hpa = int(
                item["pressure_level_hpa"]
            )
            level_index = resolve_upper_level_index(
                level_hpa
            )

            source_channel = upper[
                UPPER_VARIABLE_INDEX[variable],
                level_index,
            ]

            feature_map = resample_weather_channel(
                source_channel,
                source_rows,
                source_cols,
            )

        elif source == "vegetation":
            if variable not in vegetation:
                raise KeyError(
                    f"植被数据中找不到变量：{variable}"
                )

            feature_map = vegetation[variable]

        elif source == "population":
            feature_map = population

        else:
            raise ValueError(
                f"不支持的特征来源：{source}"
            )

        feature_map = np.nan_to_num(
            feature_map,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

        x[:, feature_index] = feature_map[mask]

        if feature_cube is not None:
            feature_cube[
                feature_index,
                mask,
            ] = feature_map[mask]

        print(
            "    [{:02d}/{:02d}] channel={} source={} "
            "variable={} level={}".format(
                feature_index + 1,
                feature_count,
                item.get("channel"),
                source,
                variable,
                item.get("pressure_level_hpa"),
            )
        )

    return x, feature_cube


# =============================================================================
# 7. 模型预测
# =============================================================================
def predict_probability_batched(
    model,
    x: np.ndarray,
) -> np.ndarray:
    probabilities = np.empty(
        x.shape[0],
        dtype=np.float32,
    )

    for start in range(
        0,
        x.shape[0],
        PREDICT_BATCH_SIZE,
    ):
        end = min(
            start + PREDICT_BATCH_SIZE,
            x.shape[0],
        )

        batch = x[start:end]

        # Pipeline 会自动执行训练时保存的：
        # imputer -> StandardScaler -> classifier
        if hasattr(model, "predict_proba"):
            batch_probability = (
                model.predict_proba(batch)[:, 1]
            )

        elif hasattr(model, "decision_function"):
            score = model.decision_function(batch)
            batch_probability = (
                1.0 / (1.0 + np.exp(-score))
            )

        else:
            batch_probability = model.predict(
                batch
            )

        probabilities[start:end] = np.asarray(
            batch_probability,
            dtype=np.float32,
        )

    return probabilities


# =============================================================================
# 8. 主流程
# =============================================================================
def main() -> None:
    dates = list(
        date_range(START_DATE, END_DATE)
    )

    (
        target_crs,
        target_transform,
        target_height,
        target_width,
    ) = load_target_grid()

    if target_crs.to_epsg() != 4326:
        raise ValueError(
            "当前 Pangu 经纬度插值要求目标网格为 EPSG:4326，"
            f"当前为 {target_crs}"
        )

    mask = load_mask(
        target_height,
        target_width,
    )

    source_rows, source_cols = (
        build_pangu_source_indices(
            target_transform,
            target_height,
            target_width,
        )
    )

    vegetation_index = build_vegetation_index()

    # 人口是静态特征，只匹配一次。
    print("[INFO] Matching population once...")
    population = load_population(
        target_crs,
        target_transform,
        target_height,
        target_width,
    )
    population[~mask] = OUTSIDE_VALUE

    (
        models,
        thresholds,
        selected_metadata,
    ) = load_models_and_configuration()

    print(
        "[INFO] Selected feature count:",
        len(selected_metadata),
    )

    ensemble_dir = (
        PREDICTION_ROOT / "ensemble"
    )
    ensemble_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_SELECTED_FEATURES:
        SELECTED_FEATURE_OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    for date_text in dates:
        print("\n" + "=" * 70)
        print("[DATE]", date_text)

        ensemble_probability_path = (
            ensemble_dir
            / f"mean_probability_{date_text}.npy"
        )
        ensemble_binary_path = (
            ensemble_dir
            / f"majority_vote_{date_text}.npy"
        )

        if (
            not OVERWRITE
            and ensemble_probability_path.exists()
            and ensemble_binary_path.exists()
        ):
            print("[SKIP] Ensemble outputs already exist.")
            continue

        upper, surface = load_pangu_weather(
            date_text
        )

        (
            matched_vegetation_date,
            vegetation_path,
        ) = nearest_vegetation_file(
            date_text,
            vegetation_index,
        )

        print(
            "  vegetation:",
            vegetation_path,
            f"(matched date={matched_vegetation_date})",
        )

        vegetation = load_vegetation(
            vegetation_path,
            target_crs,
            target_transform,
            target_height,
            target_width,
        )

        for array in vegetation.values():
            array[~mask] = OUTSIDE_VALUE

        print("  Building model input...")

        x, feature_cube = build_selected_feature_matrix(
            selected_metadata=selected_metadata,
            upper=upper,
            surface=surface,
            vegetation=vegetation,
            population=population,
            mask=mask,
            source_rows=source_rows,
            source_cols=source_cols,
        )

        print(
            "  Model input shape:",
            x.shape,
        )

        if feature_cube is not None:
            feature_path = (
                SELECTED_FEATURE_OUTPUT_DIR
                / f"selected_features_{date_text}.npy"
            )
            np.save(
                feature_path,
                feature_cube,
            )
            print(
                "  Saved selected features:",
                feature_path,
            )

        probability_maps: List[np.ndarray] = []
        binary_maps: List[np.ndarray] = []

        for model_name in MODEL_NAMES:
            model = models[model_name]
            threshold = thresholds[model_name]
            output_dir = (
                PREDICTION_ROOT / model_name
            )

            probability_path = (
                output_dir
                / f"fire_probability_{date_text}.npy"
            )
            binary_path = (
                output_dir
                / f"fire_binary_{date_text}.npy"
            )
            summary_path = (
                output_dir
                / f"prediction_summary_{date_text}.json"
            )

            probability_vector = (
                predict_probability_batched(
                    model,
                    x,
                )
            )

            binary_vector = (
                probability_vector >= threshold
            ).astype(np.uint8)

            probability_map = np.zeros(
                mask.shape,
                dtype=np.float32,
            )
            binary_map = np.zeros(
                mask.shape,
                dtype=np.uint8,
            )

            probability_map[mask] = (
                probability_vector
            )
            binary_map[mask] = binary_vector

            np.save(
                probability_path,
                probability_map,
            )
            np.save(
                binary_path,
                binary_map,
            )

            save_json(
                summary_path,
                {
                    "date": date_text,
                    "weather_source": "Pangu-Weather global NPY",
                    "upper_file": str(
                        PANGU_NPY_DIR
                        / UPPER_PATTERN.format(
                            date=date_text
                        )
                    ),
                    "surface_file": str(
                        PANGU_NPY_DIR
                        / SURFACE_PATTERN.format(
                            date=date_text
                        )
                    ),
                    "vegetation_file": str(
                        vegetation_path
                    ),
                    "matched_vegetation_date": (
                        matched_vegetation_date
                    ),
                    "model_name": model_name,
                    "threshold": float(threshold),
                    "selected_feature_count": int(
                        x.shape[1]
                    ),
                    "guangdong_pixel_count": int(
                        mask.sum()
                    ),
                    "predicted_fire_pixel_count": int(
                        binary_vector.sum()
                    ),
                    "mean_probability": float(
                        probability_vector.mean()
                    ),
                    "maximum_probability": float(
                        probability_vector.max()
                    ),
                },
            )

            probability_maps.append(
                probability_map
            )
            binary_maps.append(
                binary_map
            )

            print(
                "  {}: threshold={:.6f}, "
                "fire_pixels={}, mean_prob={:.6f}, "
                "max_prob={:.6f}".format(
                    model_name,
                    threshold,
                    int(binary_vector.sum()),
                    float(probability_vector.mean()),
                    float(probability_vector.max()),
                )
            )

        probability_stack = np.stack(
            probability_maps,
            axis=0,
        )

        binary_stack = np.stack(
            binary_maps,
            axis=0,
        )

        mean_probability = (
            probability_stack.mean(axis=0)
            .astype(np.float32)
        )

        # 严格多数：4 个模型中至少 3 个预测为火灾。
        majority_vote = (
            binary_stack.sum(axis=0)
            >= (len(MODEL_NAMES) // 2 + 1)
        ).astype(np.uint8)

        np.save(
            ensemble_probability_path,
            mean_probability,
        )
        np.save(
            ensemble_binary_path,
            majority_vote,
        )

        save_json(
            ensemble_dir
            / f"prediction_summary_{date_text}.json",
            {
                "date": date_text,
                "weather_source": "Pangu-Weather global NPY",
                "models": MODEL_NAMES,
                "probability_ensemble": (
                    "arithmetic mean"
                ),
                "binary_ensemble": (
                    "strict majority vote"
                ),
                "strict_majority_required": (
                    len(MODEL_NAMES) // 2 + 1
                ),
                "predicted_fire_pixel_count": int(
                    majority_vote[mask].sum()
                ),
                "mean_probability": float(
                    mean_probability[mask].mean()
                ),
                "maximum_probability": float(
                    mean_probability[mask].max()
                ),
            },
        )

        print(
            "  ensemble: fire_pixels={}, "
            "mean_prob={:.6f}, max_prob={:.6f}".format(
                int(majority_vote[mask].sum()),
                float(
                    mean_probability[mask].mean()
                ),
                float(
                    mean_probability[mask].max()
                ),
            )
        )

        del upper
        del surface
        del vegetation
        del x
        del feature_cube
        del probability_maps
        del binary_maps
        del probability_stack
        del binary_stack
        del mean_probability
        del majority_vote
        gc.collect()

    print("\n[DONE]")
    print(
        "Prediction root:",
        PREDICTION_ROOT.resolve(),
    )


if __name__ == "__main__":
    main()