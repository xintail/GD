#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用于保存和加载后校准火灾分类模型。

此文件应放在项目的：
    ./model/calibrated_probability_model.py

这样 checkpoint_calibration/<model_name>/model.joblib 可以被现有预测脚本
直接 joblib.load()，无需修改预测接口。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


PROBABILITY_EPS = 1e-7


def _as_probability_vector(probability) -> np.ndarray:
    array = np.asarray(probability, dtype=np.float64).reshape(-1)
    array = np.nan_to_num(
        array,
        nan=0.5,
        posinf=1.0 - PROBABILITY_EPS,
        neginf=PROBABILITY_EPS,
    )
    return np.clip(array, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def _probability_to_logit(probability: np.ndarray) -> np.ndarray:
    probability = _as_probability_vector(probability)
    return np.log(
        probability / (1.0 - probability)
    ).reshape(-1, 1)


class SigmoidProbabilityCalibrator(BaseEstimator):
    """
    对基础模型输出概率的 logit 进行逻辑回归校准。

    calibrated_probability =
        sigmoid(intercept + slope * logit(raw_probability))
    """

    def __init__(
        self,
        regularization_c: float = 1e6,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.regularization_c = regularization_c
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(
        self,
        probability,
        y,
        sample_weight=None,
    ):
        probability = _as_probability_vector(probability)
        y = np.asarray(y, dtype=np.uint8).reshape(-1)

        if probability.size != y.size:
            raise ValueError(
                "probability 与 y 的长度不一致。"
            )

        if np.unique(y).size < 2:
            raise ValueError(
                "校准数据必须同时包含正类和负类。"
            )

        self.model_ = LogisticRegression(
            C=float(self.regularization_c),
            solver="lbfgs",
            max_iter=int(self.max_iter),
            random_state=int(self.random_state),
        )

        self.model_.fit(
            _probability_to_logit(probability),
            y,
            sample_weight=sample_weight,
        )

        self.classes_ = np.array([0, 1], dtype=np.uint8)
        return self

    def predict(self, probability) -> np.ndarray:
        check_is_fitted(self, "model_")

        calibrated = self.model_.predict_proba(
            _probability_to_logit(probability)
        )[:, 1]

        return np.clip(
            calibrated,
            0.0,
            1.0,
        ).astype(np.float64)

    def predict_proba(self, probability) -> np.ndarray:
        positive = self.predict(probability)
        return np.column_stack(
            [1.0 - positive, positive]
        )


class IsotonicProbabilityCalibrator(BaseEstimator):
    """单调非参数概率校准器。"""

    def __init__(self):
        pass

    def fit(
        self,
        probability,
        y,
        sample_weight=None,
    ):
        probability = _as_probability_vector(probability)
        y = np.asarray(y, dtype=np.uint8).reshape(-1)

        if probability.size != y.size:
            raise ValueError(
                "probability 与 y 的长度不一致。"
            )

        if np.unique(y).size < 2:
            raise ValueError(
                "校准数据必须同时包含正类和负类。"
            )

        self.model_ = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )

        self.model_.fit(
            probability,
            y,
            sample_weight=sample_weight,
        )

        self.classes_ = np.array([0, 1], dtype=np.uint8)
        return self

    def predict(self, probability) -> np.ndarray:
        check_is_fitted(self, "model_")

        calibrated = self.model_.predict(
            _as_probability_vector(probability)
        )

        return np.clip(
            calibrated,
            0.0,
            1.0,
        ).astype(np.float64)

    def predict_proba(self, probability) -> np.ndarray:
        positive = self.predict(probability)
        return np.column_stack(
            [1.0 - positive, positive]
        )


def build_probability_calibrator(
    method: str,
    random_state: int = 42,
):
    method = str(method).lower()

    if method == "sigmoid":
        return SigmoidProbabilityCalibrator(
            random_state=random_state
        )

    if method == "isotonic":
        return IsotonicProbabilityCalibrator()

    raise ValueError(
        f"不支持的校准方法：{method!r}；"
        "可选 sigmoid 或 isotonic。"
    )


class CalibratedProbabilityModel(
    BaseEstimator,
    ClassifierMixin,
):
    """
    将已训练的基础模型与一维概率校准器封装为一个分类模型。

    对外接口仍是：
        predict_proba(X)
        predict(X)

    因此原预测脚本只需要把 CHECKPOINT_ROOT 从：
        ./checkpoint
    切换为：
        ./checkpoint_calibration
    """

    def __init__(
        self,
        base_model,
        calibrator,
        calibration_method: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.base_model = base_model
        self.calibrator = calibrator
        self.calibration_method = calibration_method
        self.metadata = metadata

    def fit(self, X=None, y=None):
        raise RuntimeError(
            "CalibratedProbabilityModel 是已拟合封装模型，"
            "不能再次调用 fit()。"
        )

    @property
    def classes_(self):
        return np.array([0, 1], dtype=np.uint8)

    @property
    def n_features_in_(self):
        if hasattr(self.base_model, "n_features_in_"):
            return self.base_model.n_features_in_
        raise AttributeError(
            "基础模型没有 n_features_in_。"
        )

    @property
    def feature_names_in_(self):
        if hasattr(self.base_model, "feature_names_in_"):
            return self.base_model.feature_names_in_
        raise AttributeError(
            "基础模型没有 feature_names_in_。"
        )

    def _raw_positive_probability(self, X) -> np.ndarray:
        if hasattr(self.base_model, "predict_proba"):
            raw = self.base_model.predict_proba(X)[:, 1]

        elif hasattr(self.base_model, "decision_function"):
            score = np.asarray(
                self.base_model.decision_function(X),
                dtype=np.float64,
            )
            raw = 1.0 / (1.0 + np.exp(-score))

        else:
            raw = self.base_model.predict(X)

        return _as_probability_vector(raw)

    def predict_proba(self, X) -> np.ndarray:
        raw = self._raw_positive_probability(X)
        calibrated = self.calibrator.predict(raw)

        return np.column_stack(
            [1.0 - calibrated, calibrated]
        )

    def predict(self, X) -> np.ndarray:
        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(np.uint8)

    def decision_function(self, X) -> np.ndarray:
        probability = np.clip(
            self.predict_proba(X)[:, 1],
            PROBABILITY_EPS,
            1.0 - PROBABILITY_EPS,
        )
        return np.log(
            probability / (1.0 - probability)
        )