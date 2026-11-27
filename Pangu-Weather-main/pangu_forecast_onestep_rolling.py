#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pangu-Weather 24h inference for a user-specified date range.

功能：
1) 用户指定 start_date 和 end_date，预测结果保存到 end_date（包含 end_date）。
2) 保留 upper 和 surface 输入接口：START_UPPER_PATH / START_SURFACE_PATH，二者必须与 start_date 对应。
3) 同时生成两类预测：
   - 一步预测 Pred：每个目标日 d 使用 d-1 日的外部/真实气象场作为输入，运行一次 24h 模型。
   - 滚动预测 Pred_sbs：只使用 start_date 的输入场起报，之后不断将前一天预测结果作为下一天输入。
4) 输出分别保存到两个文件夹，文件名兼容后续火灾概率绘图脚本：
   - Pangu_pred_npy/input_upper_YYYYMMDD.npy
   - Pangu_pred_npy/input_surface_YYYYMMDD.npy
   - Pangu_pred_sbs_npy/input_upper_YYYYMMDD.npy
   - Pangu_pred_sbs_npy/input_surface_YYYYMMDD.npy

重要说明：
- pangu_weather_24.onnx 是 24 小时预报模型。因此，如果 START_UPPER_PATH / START_SURFACE_PATH
  对应 start_date，则第一次模型输出对应 start_date + 1 day。
- 若要完整生成多天的一步预测，必须提供 ONE_STEP_INPUT_DIR 或命令行 --one-step-input-dir，
  其中包含每个起报日的 input_upper_YYYYMMDD.npy 和 input_surface_YYYYMMDD.npy。
  例如预测 20250102-20250131，则一步预测需要 20250101-20250130 的起报场。
- 如果不提供 ONE_STEP_INPUT_DIR，脚本仍会保存第一个 24h 一步预测；后续日期的一步预测会跳过，
  因为没有独立起报场，不能严格称为“一步预测”。
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# =============================================================================
# Windows GPU DLL setup
# =============================================================================
# This script uses the CUDA/cuDNN DLLs bundled with the installed PyTorch.
# The DLL search path must be configured BEFORE importing onnxruntime.
import torch

TORCH_LIB_DIR = os.path.join(os.path.dirname(torch.__file__), "lib")
_TORCH_DLL_DIR_HANDLE = None

if os.name == "nt" and os.path.isdir(TORCH_LIB_DIR):
    # Keep the returned handle alive for the whole process. Closing or garbage
    # collecting it removes the directory from Python's DLL search path.
    if hasattr(os, "add_dll_directory"):
        _TORCH_DLL_DIR_HANDLE = os.add_dll_directory(TORCH_LIB_DIR)
    os.environ["PATH"] = TORCH_LIB_DIR + os.pathsep + os.environ.get("PATH", "")

# onnxruntime must be imported only after the DLL path is prepared.
import onnxruntime as ort

try:
    import onnx  # only used for optional model checking
except Exception:  # pragma: no cover
    onnx = None


# =============================================================================
# 0) CONFIG: 你主要改这里，也可以用命令行覆盖
# =============================================================================
START_DATE = "20261119"
END_DATE = "20261231"

# 这两个文件必须与 START_DATE 对应。
# shape 通常为：upper=(5,13,H,W)，surface=(4,H,W)
START_SURFACE_PATH = "./Pangu-Weather-main/era5_npy/input_surface_20261119.npy"
START_UPPER_PATH = "./Pangu-Weather-main/era5_npy/input_upper_20261119.npy"

# Pangu-Weather 24h ONNX 模型
MODEL_24_PATH = r"D:\GD\v4_2026年国庆预测\Pangu-Weather-main\pangu_weather_24.onnx"

# 输出根目录。默认会在这里创建 Pangu_pred_npy 和 Pangu_pred_sbs_npy 两个文件夹。
# 若你希望直接服务于之前的绘图脚本，可以设为：../v1/data/match_data_Africa
OUTPUT_ROOT = "./"
PRED_DIR_NAME = "Pangu_pred_npy"          # 一步预测输出文件夹
PRED_SBS_DIR_NAME = "Pangu_pred_sbs_npy"  # 滚动预测输出文件夹

# 严格的一步预测需要每个起报日的真实/外部气象场。
# 文件命名要求：input_upper_YYYYMMDD.npy 和 input_surface_YYYYMMDD.npy
# 若没有，可保持 None；此时一步预测只会生成 start_date+1 这一天。
ONE_STEP_INPUT_DIR = "./data/era5_npy"  # 例如："../v1/data/match_data_Africa/era5_npy"

# 是否把 start_date 本身的输入场也复制保存到两个输出文件夹中。
# 注意：这不是模型预测结果，只是为了让后续脚本能读取完整日期序列。
SAVE_INITIAL_STATE = False

# ONNXRuntime 设置
PROVIDER = "CUDAExecutionProvider"  # 可改为 "CUDAExecutionProvider"，如果环境支持
INTRA_OP_NUM_THREADS = 1
CHECK_ONNX_MODEL = False

# 如果 ONE_STEP_INPUT_DIR 中缺少某个起报日文件："error" 或 "skip"
MISSING_ONE_STEP = "error"


# =============================================================================
# 1) 日期与路径工具
# =============================================================================
def parse_yyyymmdd(s: str) -> datetime:
    try:
        return datetime.strptime(s, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"Date must be YYYYMMDD, got {s!r}") from exc


def fmt_date(d: datetime) -> str:
    return d.strftime("%Y%m%d")


def daterange_inclusive(start: datetime, end: datetime) -> Iterable[datetime]:
    if end < start:
        raise ValueError(f"end_date {fmt_date(end)} is earlier than start_date {fmt_date(start)}")
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def target_dates_from_start(start_date: str, end_date: str) -> List[str]:
    """
    With a 24h model, input at start_date produces the first target at start_date+1.
    Return all model-output dates up to and including end_date.
    """
    start = parse_yyyymmdd(start_date)
    end = parse_yyyymmdd(end_date)
    first_target = start + timedelta(days=1)
    if end < first_target:
        return []
    return [fmt_date(d) for d in daterange_inclusive(first_target, end)]


def previous_date(date_yyyymmdd: str) -> str:
    return fmt_date(parse_yyyymmdd(date_yyyymmdd) - timedelta(days=1))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def upper_name(date_yyyymmdd: str) -> str:
    return f"input_upper_{date_yyyymmdd}.npy"


def surface_name(date_yyyymmdd: str) -> str:
    return f"input_surface_{date_yyyymmdd}.npy"


def output_paths(out_dir: str, date_yyyymmdd: str) -> Tuple[str, str]:
    return (
        os.path.join(out_dir, upper_name(date_yyyymmdd)),
        os.path.join(out_dir, surface_name(date_yyyymmdd)),
    )


def find_dated_input_files(input_dir: str, date_yyyymmdd: str) -> Tuple[str, str]:
    """Find input_upper_YYYYMMDD.npy and input_surface_YYYYMMDD.npy in input_dir."""
    upper_path = os.path.join(input_dir, upper_name(date_yyyymmdd))
    surface_path = os.path.join(input_dir, surface_name(date_yyyymmdd))
    if not os.path.exists(upper_path) or not os.path.exists(surface_path):
        raise FileNotFoundError(
            "Missing one-step initialization files for date " + date_yyyymmdd + ":\n"
            f"  upper:   {upper_path}\n"
            f"  surface: {surface_path}"
        )
    return upper_path, surface_path


# =============================================================================
# 2) ONNX session 与数据接口
# =============================================================================
def make_session(model_path: str, provider: str, intra_op_num_threads: int) -> ort.InferenceSession:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Cannot find ONNX model: " + model_path)

    if CHECK_ONNX_MODEL and onnx is not None:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("[OK] ONNX model checked:", model_path)

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = int(intra_op_num_threads)

    available = ort.get_available_providers()
    print("[INFO] PyTorch:", torch.__version__)
    print("[INFO] PyTorch CUDA:", torch.version.cuda)
    print("[INFO] torch.cuda.is_available():", torch.cuda.is_available())
    print("[INFO] PyTorch DLL directory:", TORCH_LIB_DIR)
    print("[INFO] ONNX Runtime:", ort.__version__)
    print("[INFO] ONNX Runtime providers:", available)

    if provider == "CUDAExecutionProvider" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAExecutionProvider was requested, but PyTorch cannot access CUDA. "
            "Check the NVIDIA driver and the active Python environment."
        )

    if provider not in available:
        print(f"[WARN] Provider {provider!r} is not available. Available providers: {available}")
        print("       Falling back to CPUExecutionProvider.")
        provider = "CPUExecutionProvider"

    try:
        session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=[provider],
        )
    except Exception as exc:
        if provider == "CUDAExecutionProvider":
            raise RuntimeError(
                "Failed to initialize ONNX Runtime CUDAExecutionProvider. "
                "The PyTorch DLL directory was added before importing onnxruntime: "
                f"{TORCH_LIB_DIR}. Original error: {exc}"
            ) from exc
        raise

    print("[INFO] Session providers:", session.get_providers())
    return session



def load_state(upper_path: str, surface_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(upper_path):
        raise FileNotFoundError("Cannot find upper input: " + upper_path)
    if not os.path.exists(surface_path):
        raise FileNotFoundError("Cannot find surface input: " + surface_path)

    upper = np.load(upper_path).astype(np.float32, copy=False)
    surface = np.load(surface_path).astype(np.float32, copy=False)
    validate_state(upper, surface, source=f"{upper_path} | {surface_path}")
    return upper, surface


def validate_state(upper: np.ndarray, surface: np.ndarray, source: str = "") -> None:
    if upper.ndim != 4 or upper.shape[0] != 5 or upper.shape[1] != 13:
        raise ValueError(f"upper expected shape (5,13,H,W), got {upper.shape}. Source: {source}")
    if surface.ndim != 3 or surface.shape[0] != 4:
        raise ValueError(f"surface expected shape (4,H,W), got {surface.shape}. Source: {source}")
    if upper.shape[-2:] != surface.shape[-2:]:
        raise ValueError(
            f"Grid mismatch: upper spatial shape {upper.shape[-2:]} vs surface {surface.shape[-2:]}. Source: {source}"
        )


def run_24h(session: ort.InferenceSession, upper: np.ndarray, surface: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run Pangu 24h forecast. Input names follow the original inference script."""
    output_upper, output_surface = session.run(None, {"input": upper, "input_surface": surface})
    output_upper = np.asarray(output_upper, dtype=np.float32)
    output_surface = np.asarray(output_surface, dtype=np.float32)
    validate_state(output_upper, output_surface, source="onnx_output")
    return output_upper, output_surface


def save_state(out_dir: str, date_yyyymmdd: str, upper: np.ndarray, surface: np.ndarray) -> Tuple[str, str]:
    ensure_dir(out_dir)
    upper_path, surface_path = output_paths(out_dir, date_yyyymmdd)
    np.save(upper_path, upper.astype(np.float32, copy=False))
    np.save(surface_path, surface.astype(np.float32, copy=False))
    return upper_path, surface_path


# =============================================================================
# 3) 预测逻辑
# =============================================================================
def run_rolling_forecast(
    session: ort.InferenceSession,
    start_date: str,
    end_date: str,
    start_upper_path: str,
    start_surface_path: str,
    out_dir: str,
    save_initial_state: bool = False,
) -> List[Dict[str, str]]:
    """
    Rolling / step-by-step forecast:
    start_date state -> start_date+1 forecast -> use forecast as next input -> ... -> end_date.
    """
    ensure_dir(out_dir)
    target_dates = target_dates_from_start(start_date, end_date)
    upper, surface = load_state(start_upper_path, start_surface_path)

    logs: List[Dict[str, str]] = []
    if save_initial_state:
        up, sp = save_state(out_dir, start_date, upper, surface)
        logs.append({"mode": "rolling_initial", "date": start_date, "upper": up, "surface": sp})

    for target_date in target_dates:
        upper, surface = run_24h(session, upper, surface)
        up, sp = save_state(out_dir, target_date, upper, surface)
        logs.append({"mode": "rolling", "date": target_date, "upper": up, "surface": sp})
        print(f"[OK][rolling] saved {target_date}: {up}, {sp}")

    return logs


def run_one_step_forecast(
    session: ort.InferenceSession,
    start_date: str,
    end_date: str,
    start_upper_path: str,
    start_surface_path: str,
    out_dir: str,
    one_step_input_dir: Optional[str] = None,
    save_initial_state: bool = False,
    missing: str = "error",
) -> List[Dict[str, str]]:
    """
    Strict one-step forecast:
    For each target date d, load the external/observed initialization field at d-1,
    run the 24h model once, and save the output as date d.

    For d = start_date + 1, this function always uses START_UPPER_PATH / START_SURFACE_PATH.
    For later d, it needs ONE_STEP_INPUT_DIR/input_upper_{d-1}.npy and input_surface_{d-1}.npy.
    """
    if missing not in {"error", "skip"}:
        raise ValueError("missing must be 'error' or 'skip'")

    ensure_dir(out_dir)
    target_dates = target_dates_from_start(start_date, end_date)
    logs: List[Dict[str, str]] = []

    if save_initial_state:
        upper0, surface0 = load_state(start_upper_path, start_surface_path)
        up, sp = save_state(out_dir, start_date, upper0, surface0)
        logs.append({"mode": "one_step_initial", "date": start_date, "upper": up, "surface": sp})

    if not target_dates:
        return logs

    for target_date in target_dates:
        init_date = previous_date(target_date)

        if init_date == start_date:
            init_upper_path, init_surface_path = start_upper_path, start_surface_path
        else:
            if one_step_input_dir is None:
                msg = (
                    f"[WARN][one-step] skip target {target_date}: no ONE_STEP_INPUT_DIR, "
                    f"so independent initialization for {init_date} is unavailable."
                )
                print(msg)
                logs.append({"mode": "one_step_skipped", "date": target_date, "reason": msg})
                continue
            try:
                init_upper_path, init_surface_path = find_dated_input_files(one_step_input_dir, init_date)
            except FileNotFoundError as exc:
                if missing == "skip":
                    print(f"[WARN][one-step] skip target {target_date}: {exc}")
                    logs.append({"mode": "one_step_skipped", "date": target_date, "reason": str(exc)})
                    continue
                raise

        upper, surface = load_state(init_upper_path, init_surface_path)
        pred_upper, pred_surface = run_24h(session, upper, surface)
        up, sp = save_state(out_dir, target_date, pred_upper, pred_surface)
        logs.append(
            {
                "mode": "one_step",
                "init_date": init_date,
                "date": target_date,
                "init_upper": init_upper_path,
                "init_surface": init_surface_path,
                "upper": up,
                "surface": sp,
            }
        )
        print(f"[OK][one-step] init {init_date} -> target {target_date}: {up}, {sp}")

    return logs


def save_run_metadata(path: str, cfg: Dict[str, object], logs: List[Dict[str, str]]) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {"config": cfg, "logs": logs}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[OK] saved metadata:", path)


# =============================================================================
# 4) CLI
# =============================================================================
@dataclass
class Config:
    start_date: str = START_DATE
    end_date: str = END_DATE
    start_upper_path: str = START_UPPER_PATH
    start_surface_path: str = START_SURFACE_PATH
    model_24_path: str = MODEL_24_PATH
    output_root: str = OUTPUT_ROOT
    pred_dir_name: str = PRED_DIR_NAME
    pred_sbs_dir_name: str = PRED_SBS_DIR_NAME
    one_step_input_dir: Optional[str] = ONE_STEP_INPUT_DIR
    save_initial_state: bool = SAVE_INITIAL_STATE
    provider: str = PROVIDER
    intra_op_num_threads: int = INTRA_OP_NUM_THREADS
    missing_one_step: str = MISSING_ONE_STEP

    @property
    def pred_dir(self) -> str:
        return os.path.join(self.output_root, self.pred_dir_name)

    @property
    def pred_sbs_dir(self) -> str:
        return os.path.join(self.output_root, self.pred_sbs_dir_name)


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Run Pangu-Weather 24h one-step and rolling forecasts for a date range."
    )
    p.add_argument("--start-date", default=START_DATE, help="Initial state date, YYYYMMDD. The upper/surface files correspond to this date.")
    p.add_argument("--end-date", default=END_DATE, help="Last forecast target date, YYYYMMDD, included.")
    p.add_argument("--input-upper", default=START_UPPER_PATH, help="Upper-air input .npy for start-date, shape (5,13,H,W).")
    p.add_argument("--input-surface", default=START_SURFACE_PATH, help="Surface input .npy for start-date, shape (4,H,W).")
    p.add_argument("--model", default=MODEL_24_PATH, help="Path to pangu_weather_24.onnx.")
    p.add_argument("--output-root", default=OUTPUT_ROOT, help="Root directory where the two output folders are created.")
    p.add_argument("--pred-dir-name", default=PRED_DIR_NAME, help="Folder name for one-step outputs.")
    p.add_argument("--pred-sbs-dir-name", default=PRED_SBS_DIR_NAME, help="Folder name for rolling outputs.")
    p.add_argument("--one-step-input-dir", default=ONE_STEP_INPUT_DIR, help="Directory containing dated external init fields for strict one-step forecasts.")
    p.add_argument("--save-initial", action="store_true", default=SAVE_INITIAL_STATE, help="Also save the start-date input state into both output folders.")
    p.add_argument("--provider", default=PROVIDER, help="ONNXRuntime provider, e.g. CPUExecutionProvider or CUDAExecutionProvider.")
    p.add_argument("--threads", type=int, default=INTRA_OP_NUM_THREADS, help="ONNXRuntime intra_op_num_threads.")
    p.add_argument("--missing-one-step", choices=["error", "skip"], default=MISSING_ONE_STEP, help="Behavior when a dated one-step initialization file is missing.")

    a = p.parse_args()
    return Config(
        start_date=a.start_date,
        end_date=a.end_date,
        start_upper_path=a.input_upper,
        start_surface_path=a.input_surface,
        model_24_path=a.model,
        output_root=a.output_root,
        pred_dir_name=a.pred_dir_name,
        pred_sbs_dir_name=a.pred_sbs_dir_name,
        one_step_input_dir=a.one_step_input_dir,
        save_initial_state=bool(a.save_initial),
        provider=a.provider,
        intra_op_num_threads=int(a.threads),
        missing_one_step=a.missing_one_step,
    )


def main() -> None:
    cfg = parse_args()

    # Validate dates early and make the inclusive target list explicit.
    target_dates = target_dates_from_start(cfg.start_date, cfg.end_date)
    print("[INFO] start_date:", cfg.start_date)
    print("[INFO] end_date:  ", cfg.end_date, "(included)")
    print("[INFO] target dates from 24h model:", target_dates if target_dates else "<none>")
    print("[INFO] one-step output dir:", cfg.pred_dir)
    print("[INFO] rolling output dir: ", cfg.pred_sbs_dir)

    ensure_dir(cfg.pred_sbs_dir)

    session = make_session(cfg.model_24_path, cfg.provider, cfg.intra_op_num_threads)

    logs: List[Dict[str, str]] = []

    # 滚动预测：Pangu_pred_sbs_npy
    logs.extend(
        run_rolling_forecast(
            session=session,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            start_upper_path=cfg.start_upper_path,
            start_surface_path=cfg.start_surface_path,
            out_dir=cfg.pred_sbs_dir,
            save_initial_state=cfg.save_initial_state,
        )
    )

    meta_path = os.path.join(cfg.output_root, "pangu_forecast_run_metadata.json")
    save_run_metadata(meta_path, asdict(cfg), logs)

    print("[DONE]")


if __name__ == "__main__":
    main()