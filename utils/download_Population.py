#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download the most suitable LandScan population layer for Guangdong.

The script requests 2026. If that annual layer is not yet available in the
Earth Engine collection, it automatically chooses the newest available year
not later than 2026 and records the actual year in the output filename and JSON
metadata.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import ee
import geopandas as gpd
import requests


# =============================================================================
# 0) CONFIG: normally only edit this section
# =============================================================================
PROJECT_ID = "genimi-from-25-12-04"

BOUNDARY_SHP = Path("./data/boundary/gd_boundary.shp")
REQUESTED_YEAR = 2026
FALLBACK_TO_LATEST_AVAILABLE = True

OUT_DIR = Path("./data/raw/Population")
NAME_PREFIX = "GD_LandScan_Population"

DATASET = "projects/sat-io/open-datasets/ORNL/LANDSCAN_GLOBAL"
BANDS = ["b1"]
SCALE = 1000

BOUNDARY_SIMPLIFY_M = 200
SKIP_EXISTING = True
HTTP_TIMEOUT_SECONDS = 600


# =============================================================================
# 1) Earth Engine initialization
# =============================================================================
def initialize_earth_engine() -> None:
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
    missing = [str(shp_path.with_suffix(s)) for s in required_suffixes
               if not shp_path.with_suffix(s).exists()]
    if missing:
        raise FileNotFoundError(
            "Guangdong Shapefile is incomplete. Missing files:\n  "
            + "\n  ".join(missing)
        )


def load_guangdong_region(shp_path: Path) -> Tuple[ee.Geometry, Tuple[float, float, float, float]]:
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
        gdf = gdf_projected.to_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    if hasattr(gdf.geometry, "union_all"):
        merged = gdf.geometry.union_all()
    else:
        merged = gdf.geometry.unary_union

    if not merged.is_valid:
        merged = merged.buffer(0)
    if merged.is_empty:
        raise ValueError("The merged Guangdong boundary is empty.")

    bounds = tuple(float(v) for v in merged.bounds)
    region = ee.Geometry(merged.__geo_interface__, proj="EPSG:4326", geodesic=False)
    return region, bounds


# =============================================================================
# 3) Select requested LandScan year or latest available fallback
# =============================================================================
def available_years(collection: ee.ImageCollection) -> List[int]:
    timestamps = collection.aggregate_array("system:time_start").getInfo()
    years = {
        datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).year
        for ms in timestamps
        if ms is not None
    }
    return sorted(years)


def select_population_image() -> Tuple[int, ee.Image, List[int]]:
    collection = ee.ImageCollection(DATASET).select(BANDS).sort("system:time_start")
    years = available_years(collection)

    if not years:
        raise RuntimeError(f"No images were found in {DATASET}")

    if REQUESTED_YEAR in years:
        selected_year = REQUESTED_YEAR
    elif FALLBACK_TO_LATEST_AVAILABLE:
        eligible_years = [year for year in years if year <= REQUESTED_YEAR]
        if not eligible_years:
            raise RuntimeError(
                f"No LandScan year <= {REQUESTED_YEAR} is available. "
                f"Available years: {years}"
            )
        selected_year = max(eligible_years)
        print(
            f"[WARN] LandScan {REQUESTED_YEAR} is unavailable; "
            f"using latest available year {selected_year}."
        )
    else:
        raise RuntimeError(
            f"LandScan {REQUESTED_YEAR} is unavailable. Available years: {years}"
        )

    start = ee.Date.fromYMD(selected_year, 1, 1)
    end = ee.Date.fromYMD(selected_year + 1, 1, 1)
    yearly = collection.filterDate(start, end)
    count = int(yearly.size().getInfo())
    if count == 0:
        raise RuntimeError(
            f"The collection reported year {selected_year}, but filterDate returned no image."
        )

    image = ee.Image(yearly.first())
    return selected_year, image, years


# =============================================================================
# 4) Download
# =============================================================================
def download_image(
    selected_year: int,
    image: ee.Image,
    region: ee.Geometry,
    all_years: List[int],
    bounds: Tuple[float, float, float, float],
) -> Path:
    out_name = f"{NAME_PREFIX}_{selected_year}_500m"
    out_path = OUT_DIR / f"{out_name}.tif"

    if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] already exists: {out_path}")
    else:
        image = image.clip(region)
        url = image.getDownloadURL(
            {
                "name": out_name,
                "region": region,
                "scale": SCALE,
                "format": "GEO_TIFF",
            }
        )

        temp_path = out_path.with_suffix(out_path.suffix + ".part")
        with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)

        os.replace(temp_path, out_path)
        print(f"[SAVED] LandScan {selected_year}: {out_path}")

    metadata = {
        "dataset": DATASET,
        "requested_year": REQUESTED_YEAR,
        "selected_year": selected_year,
        "fallback_used": selected_year != REQUESTED_YEAR,
        "available_years": all_years,
        "boundary_shapefile": str(BOUNDARY_SHP),
        "boundary_bounds_epsg4326": {
            "min_lon": bounds[0],
            "min_lat": bounds[1],
            "max_lon": bounds[2],
            "max_lat": bounds[3],
        },
        "scale_m": SCALE,
        "output_file": str(out_path),
    }
    metadata_path = OUT_DIR / f"{out_name}_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"[SAVED] metadata: {metadata_path}")

    return out_path


# =============================================================================
# 5) Main
# =============================================================================
def main() -> None:
    initialize_earth_engine()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    region, bounds = load_guangdong_region(BOUNDARY_SHP)
    print("[INFO] Guangdong boundary:", BOUNDARY_SHP)
    print(
        "[INFO] WGS84 bounds "
        f"(min_lon, min_lat, max_lon, max_lat): {bounds}"
    )

    selected_year, image, years = select_population_image()
    print("[INFO] available LandScan years:", years)
    print("[INFO] requested year:", REQUESTED_YEAR)
    print("[INFO] selected year:", selected_year)

    download_image(selected_year, image, region, years, bounds)
    print("[DONE] Population download completed.")


if __name__ == "__main__":
    main()