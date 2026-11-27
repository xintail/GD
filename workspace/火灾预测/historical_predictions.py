#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 checkpoint 中的多个模型，对每日广东特征进行像元级火灾预测。

每个模型输出
------------
predictions/<model_name>/
    fire_probability_YYYYMMDD.npy
    fire_binary_YYYYMMDD.npy
    prediction_summary_YYYYMMDD.json

集成输出
--------
predictions/ensemble/
    mean_probability_YYYYMMDD.npy
    majority_vote_YYYYMMDD.npy
    prediction_summary_YYYYMMDD.json

说明
----
- 每个模型使用训练阶段保存的 StandardScaler、缺失值处理器和分类器。
- 每个模型使用自身通过日期交叉验证选出的最大 F1 阈值。
- 仅对广东掩膜内像元预测，广东边界外保持 0。
"""

from __future__ import annotations
import sys
sys.path.append('./')
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence

import joblib
import numpy as np
from model.calibrated_probability_model import CalibratedProbabilityModel

# =============================================================================
# 0. 配置
# =============================================================================
START_DATE = "2025-12-01"
END_DATE = "2025-12-31"

FEATURE_DIR = Path("./data/processed/features_1km")
FEATURE_PATTERN = "feature_{date}.npy"

MASK_PATH = Path("./data/processed/guangdong_mask_1km.npy")

CHECKPOINT_ROOT = Path("./checkpoint")
PREDICTION_ROOT = Path("./workspace/火灾预测/historical_predictions")
# CHECKPOINT_ROOT = Path("./checkpoint_calibration")
# PREDICTION_ROOT = Path("./workspace/火灾预测/historical_predictions_calibration")

MODEL_NAMES = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
]

PREDICT_BATCH_SIZE = 100000
OVERWRITE = False


# =============================================================================
# 1. 工具
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


# =============================================================================
# 2. 主流程
# =============================================================================
def main() -> None:
    dates = list(
        date_range(START_DATE, END_DATE)
    )

    if not MASK_PATH.exists():
        raise FileNotFoundError(
            f"找不到广东掩膜：{MASK_PATH}"
        )

    mask = np.load(MASK_PATH).astype(bool)

    loaded_models = {}
    thresholds = {}
    selected_channels_by_model = {}

    for model_name in MODEL_NAMES:
        model_dir = CHECKPOINT_ROOT / model_name

        model_path = model_dir / "model.joblib"
        threshold_path = model_dir / "threshold.json"
        channel_path = model_dir / "selected_channels.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到模型：{model_path}"
            )
        if not threshold_path.exists():
            raise FileNotFoundError(
                f"找不到阈值：{threshold_path}"
            )
        if not channel_path.exists():
            raise FileNotFoundError(
                f"找不到通道配置：{channel_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)
        thresholds[model_name] = float(
            load_json(threshold_path)["threshold"]
        )
        selected_channels_by_model[model_name] = list(
            map(
                int,
                load_json(channel_path)[
                    "selected_channel_indices"
                ],
            )
        )

        (PREDICTION_ROOT / model_name).mkdir(
            parents=True,
            exist_ok=True,
        )

    ensemble_dir = PREDICTION_ROOT / "ensemble"
    ensemble_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 所有模型应使用相同通道。
    reference_channels = selected_channels_by_model[
        MODEL_NAMES[0]
    ]

    for model_name in MODEL_NAMES[1:]:
        if (
            selected_channels_by_model[model_name]
            != reference_channels
        ):
            raise ValueError(
                f"{model_name} 的选中特征通道与其他模型不一致。"
            )

    for date_text in dates:
        feature_path = (
            FEATURE_DIR
            / FEATURE_PATTERN.format(date=date_text)
        )

        if not feature_path.exists():
            print("[MISSING]", feature_path)
            continue

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
            print("[SKIP]", date_text)
            continue

        features = np.load(
            feature_path,
            mmap_mode="r",
        )

        if features.ndim != 3:
            raise ValueError(
                f"{feature_path} 应为 (C,H,W)，当前 {features.shape}"
            )
        if features.shape[1:] != mask.shape:
            raise ValueError(
                f"{feature_path} 空间 shape={features.shape[1:]}，"
                f"mask shape={mask.shape}"
            )

        x = np.asarray(
            features[
                np.asarray(reference_channels),
                :,
                :,
            ][:, mask].T,
            dtype=np.float32,
        )

        model_probability_maps: List[np.ndarray] = []
        model_binary_maps: List[np.ndarray] = []

        print("\n[DATE]", date_text)

        for model_name in MODEL_NAMES:
            model = loaded_models[model_name]
            threshold = thresholds[model_name]
            output_dir = PREDICTION_ROOT / model_name

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

            probability_vector = predict_probability_batched(
                model,
                x,
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

            probability_map[mask] = probability_vector
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
                    "model_name": model_name,
                    "threshold": threshold,
                    "guangdong_pixel_count": int(mask.sum()),
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

            model_probability_maps.append(
                probability_map
            )
            model_binary_maps.append(
                binary_map
            )

            print(
                f"  {model_name}: "
                f"threshold={threshold:.6f}, "
                f"fire_pixels={int(binary_vector.sum())}"
            )

        probability_stack = np.stack(
            model_probability_maps,
            axis=0,
        )
        binary_stack = np.stack(
            model_binary_maps,
            axis=0,
        )

        mean_probability = probability_stack.mean(
            axis=0
        ).astype(np.float32)

        # 严格多数投票：4 个模型中至少 3 个预测为火灾。
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
                "models": MODEL_NAMES,
                "ensemble_method_probability": "mean",
                "ensemble_method_binary": "majority vote",
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
            "  ensemble: fire_pixels={}".format(
                int(majority_vote[mask].sum())
            )
        )

    print("\n[DONE]")
    print("Prediction root:", PREDICTION_ROOT.resolve())


if __name__ == "__main__":
    main()