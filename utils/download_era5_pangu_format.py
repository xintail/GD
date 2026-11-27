#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    此脚本用于将下载grid数据
    运行方法 python ./data_download/download_era5_pangu_format.py --start 2026-07-26 --end 2026-07-26 --out ./data/era5_grib
"""

import os
import argparse
from datetime import datetime, timedelta
import cdsapi

# 变量顺序（仅作为请求与语义对齐；GRIB 内部会包含这些变量）
SURFACE_VARS = [
    "mean_sea_level_pressure",        # MSLP
    "10m_u_component_of_wind",        # U10
    "10m_v_component_of_wind",        # V10
    "2m_temperature",                 # T2M
]

UPPER_VARS = [
    "geopotential",                   # Z
    "specific_humidity",              # Q
    "temperature",                    # T
    "u_component_of_wind",            # U
    "v_component_of_wind",            # V
]

PRESSURE_LEVELS = ["1000","925","850","700","600","500","400","300","250","200","150","100","50"]

# 0.25° 全球网格通常对应 721 x 1440
GRID = [0.25, 0.25]
# 指定全球范围（N,W,S,E），并使用 0~360 经度以匹配 1440 经向
AREA_GLOBAL_0360 = [90, 0, -90, 359.75]

def daterange(start_date: datetime, end_date: datetime):
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def download_one_day(client: cdsapi.Client, day: datetime, out_dir: str):
    y = f"{day.year:04d}"
    m = f"{day.month:02d}"
    d = f"{day.day:02d}"
    datestr = f"{y}{m}{d}"

    out_surface = os.path.join(out_dir, f"input_surface_{datestr}.grib")
    out_upper   = os.path.join(out_dir, f"input_upper_{datestr}.grib")

    # 1) surface GRIB
    if not os.path.exists(out_surface):
        surface_request = {
            "product_type": ["reanalysis"],
            "variable": SURFACE_VARS,
            "year": y,
            "month": m,
            "day": d,
            "time": ["12:00"],     # noon UTC
            "data_format": "grib",
            "download_format": "unarchived",
            "grid": GRID,
            "area": AREA_GLOBAL_0360,
        }
        print(f"[DL] surface {datestr} ...")
        r = client.retrieve("reanalysis-era5-single-levels", surface_request)
        r.download(out_surface)
        print(f"[OK] wrote {out_surface}")

    else:
        print(f"[SKIP] surface {datestr} exists: {out_surface}")

    # # 2) upper GRIB
    # if not os.path.exists(out_upper):
    #     upper_request = {
    #         "product_type": ["reanalysis"],
    #         "variable": UPPER_VARS,
    #         "pressure_level": PRESSURE_LEVELS,
    #         "year": y,
    #         "month": m,
    #         "day": d,
    #         "time": ["12:00"],     # noon UTC
    #         "data_format": "grib",
    #         "download_format": "unarchived",
    #         "grid": GRID,
    #         "area": AREA_GLOBAL_0360,
    #     }
    #     print(f"[DL] upper {datestr} ...")
    #     r = client.retrieve("reanalysis-era5-pressure-levels", upper_request)
    #     r.download(out_upper)
    #     print(f"[OK] wrote {out_upper}")

    # else:
    #     print(f"[SKIP] upper {datestr} exists: {out_upper}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date, YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date, YYYY-MM-DD (inclusive)")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")
    os.makedirs(args.out, exist_ok=True)

    client = cdsapi.Client()

    for day in daterange(start_date, end_date):
        download_one_day(client, day, args.out)

    print("Done.")

if __name__ == "__main__":
    main()
    # python ./data_download/download_era5_pangu_format.py --start 2026-07-26 --end 2026-07-26 --out ./data/raw/era5_grib