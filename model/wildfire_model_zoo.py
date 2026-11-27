#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一定义多个 sklearn 二分类模型。

所有模型都放入 Pipeline：
    SimpleImputer -> StandardScaler -> Classifier

这样每个交叉验证折中的缺失值处理和归一化参数都只从训练日期拟合，
不会使用验证日期信息。
"""

from __future__ import annotations

from typing import Dict, List

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAMES: List[str] = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
]


def _preprocess_steps():
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


def build_model(
    model_name: str,
    random_state: int = 42,
) -> Pipeline:
    """
    根据名称创建一个全新的 sklearn Pipeline。
    """
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="saga",
            class_weight="balanced",
            max_iter=1500,
            n_jobs=-1,
            random_state=random_state,
        )

    elif model_name == "sgd_logistic":
        classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=3000,
            tol=1e-4,
            average=True,
            random_state=random_state,
        )

    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    elif model_name == "extra_trees":
        classifier = ExtraTreesClassifier(
            n_estimators=250,
            max_depth=24,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    else:
        raise ValueError(
            f"未知模型 {model_name!r}。可选模型：{MODEL_NAMES}"
        )

    return Pipeline(
        _preprocess_steps()
        + [("classifier", classifier)]
    )


def model_parameter_summary() -> Dict[str, Dict[str, object]]:
    """
    返回便于写入日志的模型参数。
    """
    summary: Dict[str, Dict[str, object]] = {}

    for name in MODEL_NAMES:
        model = build_model(name)
        summary[name] = model.get_params(deep=False)

    return summary