#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download daily FIRMS confidence rasters for Guangdong in October 2026.

The Guangdong boundary is read from:
    ./data/boundary/gd_boundary.shp
The accompanying .dbf/.prj/.shx files must be in the same directory.

Important:
    October 2026 is a future period as of 2026-08-01. By default, this script
    refuses to create future all-zero rasters when FIRMS data are not yet
    available. Run it again after the requested date range has been published.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ee
import geopandas as gpd
import requests


# =============================================================================
# 0) CONFIG: normally only edit this section
# =============================================================================
PROJECT_ID = "genimi-from-25-12-04"

BOUNDARY_SHP = Path("./data/boundary/gd_boundary.shp")

# Guangdong fire-prediction/evaluation period. END_DATE includes that day.
START_DATE = "2025-12-01"
END_DATE = "2025-12-31"

OUT_DIR = Path("./data/raw/FIRMS")
NAME_PREFIX = "GD_FIRMS_conf"

DATASET = "FIRMS"
CONFIDENCE_BAND = "confidence"

# Preserve the nominal 1 km spatial resolution of the FIRMS/MODIS source.
# In EPSG:4326, 1 km is approximately 0.008983 degrees. A fixed transform is
# still used so every daily raster has exactly the same extent and dimensions.
TARGET_CRS = "EPSG:4326"
TARGET_RESOLUTION_M = 1000
TARGET_RESOLUTION_DEG = TARGET_RESOLUTION_M / 111_319.49079327358

# Use a smaller simplification tolerance than the output pixel size so narrow
# coastal features and islands are not removed before export.
BOUNDARY_SIMPLIFY_M = 200

# Prevent unavailable future dates from being silently written as all-zero
# rasters. Keep this True for the October 2026 download.
REQUIRE_COMPLETE_DATE_RANGE = True

SKIP_EXISTING = True
HTTP_TIMEOUT_SECONDS = 600


# =============================================================================
# 1) Earth Engine initialization
# =============================================================================
def initialize_earth_engine() -> None:
    """Initialize with existing credentials; authenticate only if necessary."""
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)


# =============================================================================
# 2) Read Guangdong Shapefile and convert it to ee.Geometry
# =============================================================================
def check_shapefile_components(shp_path: Path) -> None:
    required_suffixes = [".shp", ".shx", ".dbf", ".prj"]
    missing = [
        str(shp_path.with_suffix(suffix))
        for suffix in required_suffixes
        if not shp_path.with_suffix(suffix).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Guangdong Shapefile is incomplete. Missing files:\n  "
            + "\n  ".join(missing)
        )


def load_guangdong_region(
    shp_path: Path,
) -> Tuple[ee.Geometry, Tuple[float, float, float, float]]:
    """Read, reproject, merge and simplify all Guangdong polygons."""
    check_shapefile_components(shp_path)

    gdf = gpd.read_file(shp_path)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError(f"No valid geometry was found in {shp_path}")
    if gdf.crs is None:
        raise ValueError(
            f"{shp_path.with_suffix('.prj')} did not provide a readable CRS. "
            "Please confirm that the .prj file is valid."
        )

    # Simplify in a projected CRS because the tolerance is measured in metres.
    if BOUNDARY_SIMPLIFY_M > 0:
        try:
            projected_crs = gdf.estimate_utm_crs()
        except Exception:
            projected_crs = None
        if projected_crs is None:
            projected_crs = "EPSG:3857"

        gdf_projected = gdf.to_crs(projected_crs)
        gdf_projected["geometry"] = gdf_projected.geometry.simplify(
            BOUNDARY_SIMPLIFY_M,
            preserve_topology=True,
        )
        gdf = gdf_projected.to_crs(TARGET_CRS)
    else:
        gdf = gdf.to_crs(TARGET_CRS)

    # Support both newer and older GeoPandas versions.
    if hasattr(gdf.geometry, "union_all"):
        merged = gdf.geometry.union_all()
    else:
        merged = gdf.geometry.unary_union

    if not merged.is_valid:
        merged = merged.buffer(0)
    if merged.is_empty:
        raise ValueError("The merged Guangdong boundary is empty.")

    bounds = tuple(float(value) for value in merged.bounds)
    region = ee.Geometry(
        merged.__geo_interface__,
        proj=TARGET_CRS,
        geodesic=False,
    )
    return region, bounds


# =============================================================================
# 3) Build a fixed Guangdong output grid
# =============================================================================
def build_fixed_grid(
    bounds: Tuple[float, float, float, float],
) -> Tuple[ee.Geometry, List[float], Dict[str, float]]:
    """Snap the boundary bounds to one fixed ~1 km EPSG:4326 grid.

    Every daily raster therefore has exactly the same transform, dimensions and
    extent. Pixels outside the Guangdong polygon but inside this rectangle are
    written as valid zeros, matching the original script's behavior.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    resolution = TARGET_RESOLUTION_DEG

    left = math.floor(min_lon / resolution) * resolution
    right = math.ceil(max_lon / resolution) * resolution
    bottom = math.floor(min_lat / resolution) * resolution
    top = math.ceil(max_lat / resolution) * resolution

    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))

    # Recompute right/bottom from integer dimensions to avoid floating error.
    right = left + width * resolution
    bottom = top - height * resolution

    transform = [
        resolution,
        0.0,
        left,
        0.0,
        -resolution,
        top,
    ]
    grid_region = ee.Geometry.Rectangle(
        [left, bottom, right, top],
        proj=TARGET_CRS,
        geodesic=False,
    )

    grid_info = {
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
        "resolution_m_nominal": TARGET_RESOLUTION_M,
        "resolution_degree": resolution,
        "width": width,
        "height": height,
    }
    return grid_region, transform, grid_info


# =============================================================================
# 4) Validate requested dates against FIRMS availability
# =============================================================================
def parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Date must be YYYY-MM-DD, got {value!r}") from exc


def latest_firms_date() -> datetime:
    collection = ee.ImageCollection(DATASET).sort("system:time_start", False)
    count = int(collection.size().getInfo())
    if count == 0:
        raise RuntimeError(f"No images were found in Earth Engine dataset {DATASET!r}.")

    latest_timestamp = ee.Image(collection.first()).get("system:time_start").getInfo()
    if latest_timestamp is None:
        raise RuntimeError(
            f"The latest image in {DATASET!r} has no system:time_start property."
        )

    return datetime.fromtimestamp(
        float(latest_timestamp) / 1000.0,
        tz=timezone.utc,
    ).replace(tzinfo=None)


def validate_requested_dates() -> Tuple[datetime, datetime, datetime]:
    start = parse_date(START_DATE)
    end = parse_date(END_DATE)

    if end < start:
        raise ValueError(f"END_DATE {END_DATE} is earlier than START_DATE {START_DATE}.")

    minimum_start = datetime(2010, 1, 1)
    if start < minimum_start:
        raise ValueError(
            f"FIRMS download cannot start before {minimum_start:%Y-%m-%d}; "
            f"got {START_DATE}."
        )

    latest = latest_firms_date()
    print(f"[INFO] latest FIRMS image date currently available: {latest:%Y-%m-%d}")

    if REQUIRE_COMPLETE_DATE_RANGE and latest.date() < end.date():
        raise RuntimeError(
            "The requested FIRMS date range is not yet completely available.\n"
            f"  requested: {START_DATE} through {END_DATE} (inclusive)\n"
            f"  latest available FIRMS image: {latest:%Y-%m-%d}\n"
            "The downloader stopped to avoid treating unavailable future data "
            "as real zero-fire rasters. Run it again after the October 2026 "
            "FIRMS observations have been published."
        )

    return start, end, latest


# =============================================================================
# 5) Build one daily FIRMS confidence image
# =============================================================================
def build_daily_image(
    day: ee.Date,
    firms: ee.ImageCollection,
    guangdong_region: ee.Geometry,
) -> ee.Image:
    next_day = day.advance(1, "day")
    daily = firms.filterDate(day, next_day)

    # When the date itself is available but no active fire is detected in the
    # province, the resulting raster should correctly contain zeros.
    image = ee.Image(
        ee.Algorithms.If(
            daily.size().gt(0),
            daily.select(CONFIDENCE_BAND).max(),
            ee.Image(0).rename(CONFIDENCE_BAND),
        )
    )

    confidence = image.select(CONFIDENCE_BAND)
    guangdong_mask = ee.Image.constant(1).clip(guangdong_region)

    # 1) retain only Guangdong;
    # 2) retain positive fire-confidence values;
    # 3) convert no-fire and outside-province pixels to valid zeros;
    # 4) do not clip again, so the exported rectangle keeps a fixed grid.
    return (
        confidence.updateMask(guangdong_mask)
        .updateMask(confidence.gt(0))
        .unmask(0, False)
        .toUint16()
    )


# =============================================================================
# 6) Generate daily image list
# =============================================================================
def get_images(start: datetime, end: datetime) -> List[Tuple[str, ee.Image]]:
    end_exclusive = end + timedelta(days=1)

    firms = ee.ImageCollection(DATASET).filterDate(
        start.strftime("%Y-%m-%d"),
        end_exclusive.strftime("%Y-%m-%d"),
    )

    n_days = (end - start).days + 1
    images: List[Tuple[str, ee.Image]] = []

    start_ee = ee.Date(start.strftime("%Y-%m-%d"))
    for index in range(n_days):
        day = start_ee.advance(index, "day")
        date_str = day.format("YYYYMMdd").getInfo()
        images.append((date_str, day))

    # Store ee.Date rather than a prebuilt image so the Guangdong geometry can
    # be applied later without repeatedly serializing it here.
    return [(date_str, day) for date_str, day in images]


# =============================================================================
# 7) Download one daily GeoTIFF
# =============================================================================
def download_image(
    date_str: str,
    image: ee.Image,
    grid_region: ee.Geometry,
    transform: List[float],
) -> Path:
    out_name = f"{NAME_PREFIX}_{date_str}_MODIS1km"
    out_path = OUT_DIR / f"{out_name}.tif"

    if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] already exists: {out_path}")
        return out_path

    url = image.getDownloadURL(
        {
            "name": out_name,
            "region": grid_region,
            "crs": TARGET_CRS,
            "crs_transform": transform,
            "format": "GEO_TIFF",
        }
    )

    temp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with requests.get(
            url,
            stream=True,
            timeout=HTTP_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        os.replace(temp_path, out_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    print(f"[SAVED] {date_str}: {out_path}")
    return out_path


# =============================================================================
# 8) Save grid metadata
# =============================================================================
def save_metadata(
    boundary_bounds: Tuple[float, float, float, float],
    transform: List[float],
    grid_info: Dict[str, float],
    latest_date: datetime,
) -> None:
    metadata = {
        "dataset": DATASET,
        "band": CONFIDENCE_BAND,
        "start_date_inclusive": START_DATE,
        "end_date_inclusive": END_DATE,
        "latest_firms_date_at_run": latest_date.strftime("%Y-%m-%d"),
        "boundary_shapefile": str(BOUNDARY_SHP),
        "boundary_bounds_epsg4326": {
            "min_lon": boundary_bounds[0],
            "min_lat": boundary_bounds[1],
            "max_lon": boundary_bounds[2],
            "max_lat": boundary_bounds[3],
        },
        "target_crs": TARGET_CRS,
        "target_resolution_m_nominal": TARGET_RESOLUTION_M,
        "target_transform": transform,
        "grid": grid_info,
        "outside_guangdong_value": 0,
        "no_fire_value": 0,
        "output_directory": str(OUT_DIR),
        "output_name_pattern": f"{NAME_PREFIX}_YYYYMMDD_MODIS1km.tif",
    }

    metadata_path = OUT_DIR / "firms_guangdong_202610_grid_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"[SAVED] metadata: {metadata_path}")


# =============================================================================
# 9) Main
# =============================================================================
def main() -> None:
    initialize_earth_engine()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    guangdong_region, boundary_bounds = load_guangdong_region(BOUNDARY_SHP)
    grid_region, transform, grid_info = build_fixed_grid(boundary_bounds)

    print("[INFO] Guangdong boundary:", BOUNDARY_SHP)
    print(
        "[INFO] WGS84 boundary bounds "
        f"(min_lon, min_lat, max_lon, max_lat): {boundary_bounds}"
    )
    print("[INFO] fixed output grid:", grid_info)
    print("[INFO] target transform:", transform)
    print(f"[INFO] requested dates: {START_DATE} through {END_DATE} (inclusive)")

    start, end, latest = validate_requested_dates()

    end_exclusive = (end + timedelta(days=1)).strftime("%Y-%m-%d")
    firms = ee.ImageCollection(DATASET).filterDate(
        start.strftime("%Y-%m-%d"),
        end_exclusive,
    )

    daily_dates = get_images(start, end)
    print("[INFO] total download days:", len(daily_dates))

    for date_str, day in daily_dates:
        print("[DOWNLOAD]", date_str)
        image = build_daily_image(day, firms, guangdong_region).rename(date_str)
        download_image(date_str, image, grid_region, transform)

    save_metadata(boundary_bounds, transform, grid_info, latest)
    print("[DONE] Guangdong FIRMS download completed.")


if __name__ == "__main__":
    main()