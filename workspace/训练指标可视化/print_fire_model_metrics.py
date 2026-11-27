#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 evaluate_fire_models_and_export_tables.py 已生成的 CSV 表格，
在终端打印各模型的：

1. 混淆矩阵
2. ROC AUC
3. Recall
4. Precision
5. F1

默认读取正式模型比较使用的 checkpoint OOF 结果：

    ./workspace/训练指标可视化/
        evaluate_fire_models_and_export_tables/
        tables/
        ├── checkpoint_cv_metrics.csv
        └── combined_oof_confusion_matrix_data.csv

本脚本不会重新计算指标，也不会生成图表或 CSV。

运行
----
仅打印到终端：

    python print_fire_model_metrics.py

打印到终端，并同时保存 TXT：

    python print_fire_model_metrics.py --save-txt

读取校准模型的评估结果：

    python print_fire_model_metrics.py \
        --evaluation-root "./workspace/训练指标可视化/evaluate_fire_models_and_export_tables_calibration"

指定 TXT 路径：

    python print_fire_model_metrics.py \
        --save-txt \
        --txt-path "./model_metrics.txt"

也可使用操作系统重定向：

PowerShell:
    python print_fire_model_metrics.py | Tee-Object model_metrics.txt

Windows CMD:
    python print_fire_model_metrics.py > model_metrics.txt

Linux:
    python print_fire_model_metrics.py | tee model_metrics.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


# =============================================================================
# 0. 默认配置
# =============================================================================
# EVALUATION_ROOT = Path("./workspace/训练指标可视化/evaluate_fire_models_and_export_tables")
EVALUATION_ROOT = Path("./workspace/训练指标可视化/evaluate_fire_models_and_export_tables_calibration")

TABLE_DIR_NAME = "tables"
METRICS_FILE_NAME = "checkpoint_cv_metrics.csv"
CONFUSION_FILE_NAME = "combined_oof_confusion_matrix_data.csv"
DEFAULT_TXT_NAME = "checkpoint_oof_model_metrics.txt"

PREFERRED_MODEL_ORDER = [
    "logistic_regression",
    "sgd_logistic",
    "random_forest",
    "extra_trees",
    "ensemble",
]


# =============================================================================
# 1. 参数
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取模型评估 CSV，并打印混淆矩阵、ROC AUC、"
            "Recall、Precision 和 F1。"
        )
    )

    parser.add_argument(
        "--evaluation-root",
        default=str(EVALUATION_ROOT),
        help=(
            "evaluate_fire_models_and_export_tables.py 的输出根目录。"
        ),
    )

    parser.add_argument(
        "--table-dir",
        default=None,
        help=(
            "直接指定 tables 目录。提供后将优先于 --evaluation-root。"
        ),
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="将与终端相同的内容同时保存到 TXT。",
    )

    parser.add_argument(
        "--txt-path",
        default=None,
        help=(
            "TXT 输出路径，仅在使用 --save-txt 时生效。"
        ),
    )

    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="浮点指标保留的小数位数，默认 6。",
    )

    return parser.parse_args()


# =============================================================================
# 2. CSV 读取
# =============================================================================
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"找不到表格：{path}")

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV 没有表头：{path}")

        return [
            {
                str(key): "" if value is None else str(value)
                for key, value in row.items()
            }
            for row in reader
        ]


def require_columns(
    rows: Sequence[Mapping[str, str]],
    required: Iterable[str],
    path: Path,
) -> None:
    if not rows:
        raise ValueError(f"CSV 没有数据行：{path}")

    available = set(rows[0].keys())
    missing = [
        column
        for column in required
        if column not in available
    ]

    if missing:
        raise KeyError(
            f"{path} 缺少列：{missing}；"
            f"当前列为：{sorted(available)}"
        )


def parse_float(
    row: Mapping[str, str],
    column: str,
    model_name: str,
) -> float:
    text = str(row[column]).strip()

    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(
            f"模型 {model_name} 的 {column} 不是有效数字："
            f"{text!r}"
        ) from exc


def parse_int_value(
    value: str,
    description: str,
) -> int:
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"{description} 不是有效数字：{value!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{description} 不是有限数字：{value!r}"
        )

    rounded = int(round(number))

    if not math.isclose(
        number,
        rounded,
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            f"{description} 应为整数，实际为 {number}"
        )

    return rounded


# =============================================================================
# 3. 整理指标和混淆矩阵
# =============================================================================
def ordered_model_names(
    metric_rows: Sequence[Mapping[str, str]],
) -> List[str]:
    discovered: List[str] = []

    for row in metric_rows:
        model = str(row["model"]).strip()
        if model and model not in discovered:
            discovered.append(model)

    preferred = [
        model
        for model in PREFERRED_MODEL_ORDER
        if model in discovered
    ]

    remaining = sorted(
        model
        for model in discovered
        if model not in preferred
    )

    return preferred + remaining


def build_metric_lookup(
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> Dict[str, Dict[str, float]]:
    required = [
        "model",
        "roc_auc",
        "recall",
        "precision",
        "f1",
    ]
    require_columns(rows, required, path)

    lookup: Dict[str, Dict[str, float]] = {}

    for row in rows:
        model = str(row["model"]).strip()

        if not model:
            raise ValueError(
                f"{path} 中存在空模型名称。"
            )

        if model in lookup:
            raise ValueError(
                f"{path} 中模型 {model!r} 出现多次。"
            )

        lookup[model] = {
            "roc_auc": parse_float(
                row,
                "roc_auc",
                model,
            ),
            "recall": parse_float(
                row,
                "recall",
                model,
            ),
            "precision": parse_float(
                row,
                "precision",
                model,
            ),
            "f1": parse_float(
                row,
                "f1",
                model,
            ),
        }

        # 指标表中通常也包含 TN/FP/FN/TP，用于与混淆矩阵表核对。
        for column in ["tn", "fp", "fn", "tp"]:
            if (
                column in row
                and str(row[column]).strip() != ""
            ):
                lookup[model][column] = float(
                    parse_int_value(
                        row[column],
                        f"{model}.{column}",
                    )
                )

    return lookup


def build_confusion_lookup(
    rows: Sequence[Mapping[str, str]],
    path: Path,
) -> Dict[str, Tuple[int, int, int, int]]:
    required = [
        "model",
        "true_class",
        "predicted_class",
        "count",
    ]
    require_columns(rows, required, path)

    cells: Dict[
        str,
        Dict[Tuple[int, int], int],
    ] = {}

    for row in rows:
        model = str(row["model"]).strip()

        if not model:
            raise ValueError(
                f"{path} 中存在空模型名称。"
            )

        true_class = parse_int_value(
            row["true_class"],
            f"{model}.true_class",
        )
        predicted_class = parse_int_value(
            row["predicted_class"],
            f"{model}.predicted_class",
        )
        count = parse_int_value(
            row["count"],
            (
                f"{model}.count"
                f"[true={true_class},pred={predicted_class}]"
            ),
        )

        if true_class not in {0, 1}:
            raise ValueError(
                f"{model} 的 true_class 必须为 0 或 1，"
                f"实际为 {true_class}"
            )

        if predicted_class not in {0, 1}:
            raise ValueError(
                f"{model} 的 predicted_class 必须为 0 或 1，"
                f"实际为 {predicted_class}"
            )

        key = (true_class, predicted_class)
        model_cells = cells.setdefault(model, {})

        if key in model_cells:
            raise ValueError(
                f"{path} 中 {model} 的混淆矩阵单元格 "
                f"{key} 出现多次。"
            )

        model_cells[key] = count

    lookup: Dict[
        str,
        Tuple[int, int, int, int],
    ] = {}

    expected_keys = {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }

    for model, model_cells in cells.items():
        missing = expected_keys - set(
            model_cells.keys()
        )

        if missing:
            raise ValueError(
                f"{model} 的混淆矩阵缺少单元格："
                f"{sorted(missing)}"
            )

        lookup[model] = (
            model_cells[(0, 0)],  # TN
            model_cells[(0, 1)],  # FP
            model_cells[(1, 0)],  # FN
            model_cells[(1, 1)],  # TP
        )

    return lookup


def validate_table_consistency(
    metric_lookup: Mapping[
        str,
        Mapping[str, float],
    ],
    confusion_lookup: Mapping[
        str,
        Tuple[int, int, int, int],
    ],
) -> None:
    metric_models = set(metric_lookup.keys())
    confusion_models = set(
        confusion_lookup.keys()
    )

    if metric_models != confusion_models:
        only_metrics = sorted(
            metric_models - confusion_models
        )
        only_confusion = sorted(
            confusion_models - metric_models
        )

        raise ValueError(
            "总体指标表与混淆矩阵表中的模型不一致。\n"
            f"仅在指标表中：{only_metrics}\n"
            f"仅在混淆矩阵表中：{only_confusion}"
        )

    for model, metrics in metric_lookup.items():
        if not all(
            column in metrics
            for column in ["tn", "fp", "fn", "tp"]
        ):
            continue

        matrix_values = confusion_lookup[model]
        metric_values = (
            int(metrics["tn"]),
            int(metrics["fp"]),
            int(metrics["fn"]),
            int(metrics["tp"]),
        )

        if matrix_values != metric_values:
            raise ValueError(
                f"{model} 的两个 CSV 表记录不一致：\n"
                f"混淆矩阵表={matrix_values}\n"
                f"总体指标表={metric_values}"
            )


# =============================================================================
# 4. 打印
# =============================================================================
def format_metric(
    value: float,
    decimals: int,
) -> str:
    if math.isnan(value):
        return "NaN"

    if math.isinf(value):
        return (
            "Inf"
            if value > 0
            else "-Inf"
        )

    return f"{value:.{decimals}f}"


def print_one_model(
    model: str,
    metrics: Mapping[str, float],
    confusion: Tuple[int, int, int, int],
    decimals: int,
) -> None:
    tn, fp, fn, tp = confusion

    print("=" * 72)
    print(f"模型：{model}")
    print("-" * 72)
    print("混淆矩阵（行=真实类别，列=预测类别）")
    print()
    print(
        f"{'':>18}"
        f"{'预测非火灾(0)':>20}"
        f"{'预测火灾(1)':>20}"
    )
    print(
        f"{'真实非火灾(0)':>18}"
        f"{tn:>20,}"
        f"{fp:>20,}"
    )
    print(
        f"{'真实火灾(1)':>18}"
        f"{fn:>20,}"
        f"{tp:>20,}"
    )
    print()
    print(f"TN（正确判定非火灾）: {tn:,}")
    print(f"FP（误报为火灾）    : {fp:,}")
    print(f"FN（漏报火灾）      : {fn:,}")
    print(f"TP（正确检出火灾）  : {tp:,}")
    print()
    print(
        "ROC AUC  : "
        + format_metric(
            metrics["roc_auc"],
            decimals,
        )
    )
    print(
        "Recall   : "
        + format_metric(
            metrics["recall"],
            decimals,
        )
    )
    print(
        "Precision: "
        + format_metric(
            metrics["precision"],
            decimals,
        )
    )
    print(
        "F1       : "
        + format_metric(
            metrics["f1"],
            decimals,
        )
    )
    print()


def print_report(
    table_dir: Path,
    decimals: int,
) -> None:
    metrics_path = (
        table_dir / METRICS_FILE_NAME
    )
    confusion_path = (
        table_dir / CONFUSION_FILE_NAME
    )

    metric_rows = read_csv_rows(
        metrics_path
    )
    confusion_rows = read_csv_rows(
        confusion_path
    )

    model_order = ordered_model_names(
        metric_rows
    )

    metric_lookup = build_metric_lookup(
        metric_rows,
        metrics_path,
    )
    confusion_lookup = (
        build_confusion_lookup(
            confusion_rows,
            confusion_path,
        )
    )

    validate_table_consistency(
        metric_lookup,
        confusion_lookup,
    )

    print()
    print("#" * 72)
    print("广东省火灾分类模型 OOF 评估指标")
    print("#" * 72)
    print(f"指标表：{metrics_path.resolve()}")
    print(
        "混淆矩阵表："
        f"{confusion_path.resolve()}"
    )
    print(f"模型数量：{len(model_order)}")
    print()

    for model in model_order:
        print_one_model(
            model=model,
            metrics=metric_lookup[model],
            confusion=confusion_lookup[model],
            decimals=decimals,
        )

    print("#" * 72)
    print("打印完成")
    print("#" * 72)


# =============================================================================
# 5. 主流程
# =============================================================================
def main() -> None:
    args = parse_args()

    if args.decimals < 0:
        raise ValueError(
            "--decimals 不能小于 0。"
        )

    evaluation_root = Path(
        args.evaluation_root
    )

    table_dir = (
        Path(args.table_dir)
        if args.table_dir is not None
        else evaluation_root / TABLE_DIR_NAME
    )

    if not args.save_txt:
        print_report(
            table_dir=table_dir,
            decimals=args.decimals,
        )
        return

    txt_path = (
        Path(args.txt_path)
        if args.txt_path is not None
        else evaluation_root
        / DEFAULT_TXT_NAME
    )

    buffer = io.StringIO()

    with redirect_stdout(buffer):
        print_report(
            table_dir=table_dir,
            decimals=args.decimals,
        )

    report_text = buffer.getvalue()

    # 原样输出到终端。
    sys.stdout.write(report_text)

    # 同时保存 TXT。
    txt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    txt_path.write_text(
        report_text,
        encoding="utf-8",
    )

    print(f"\nTXT 已保存：{txt_path.resolve()}")


if __name__ == "__main__":
    main()
