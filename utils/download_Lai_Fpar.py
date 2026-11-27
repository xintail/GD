#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download MODIS MCD15A3H LAI/FPAR for Guangdong around October 2026.

The Guangdong boundary is read from:
    ./data/boundary/gd_boundary.shp
The accompanying .dbf/.prj/.shx files must be in the same directory.
"""

from __future__ import annotations

import os
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

# October 2026 prediction month. A small buffer is deliberately included so
# daily matching by nearest 4-day MODIS product also works at month boundaries.
START_DATE = "2025-11-28"          # inclusive
END_DATE = "2025-12-31"            # exclusive in Earth Engine

OUT_DIR = Path("./data/raw/Fpar_Lai")
NAME_PREFIX = "GD_MCD15A3H_Fpar_Lai"

DATASET = "MODIS/061/MCD15A3H"
BANDS = ["Fpar", "Lai"]
SCALE = 500

# Simplify the local boundary before sending it to Earth Engine. At a 5 km
# output scale, 1 km simplification keeps the provincial outline while avoiding
# an unnecessarily large request geometry.
BOUNDARY_SIMPLIFY_M = 100
SKIP_EXISTING = True
HTTP_TIMEOUT_SECONDS = 600


# =============================================================================
# 1) Earth Engine initialization
# =============================================================================
def initialize_earth_engine() -> None:
    """Authenticate only when the current credentials are unavailable."""
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
    """Read all polygons, reproject to WGS84, merge, simplify and return EE geometry."""
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

    # Simplification must be performed in a projected CRS because the tolerance
    # is specified in metres.
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

    # Support both newer and older GeoPandas versions.
    if hasattr(gdf.geometry, "union_all"):
        merged = gdf.geometry.union_all()
    else:
        merged = gdf.geometry.unary_union

    if not merged.is_valid:
        merged = merged.buffer(0)
    if merged.is_empty:
        raise ValueError("The merged Guangdong boundary is empty.")

    bounds = tuple(float(v) for v in merged.bounds)  # min_lon, min_lat, max_lon, max_lat
    region = ee.Geometry(merged.__geo_interface__, proj="EPSG:4326", geodesic=False)
    return region, bounds


# =============================================================================
# 3) Query MCD15A3H 4-day products
# =============================================================================
def get_images() -> List[Tuple[str, ee.Image]]:
    collection = (
        ee.ImageCollection(DATASET)
        .filterDate(START_DATE, END_DATE)
        .select(BANDS)
        .sort("system:time_start")
    )

    count = int(collection.size().getInfo())
    if count == 0:
        latest_collection = ee.ImageCollection(DATASET).sort("system:time_start", False)
        latest_count = int(latest_collection.size().getInfo())
        latest_date = "unknown"
        if latest_count > 0:
            latest_date = ee.Date(
                ee.Image(latest_collection.first()).get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()

        raise RuntimeError(
            f"No {DATASET} images are currently available between "
            f"{START_DATE} and {END_DATE}. Latest collection date: {latest_date}. "
            "October 2026 is a future period, so run this downloader again after "
            "the corresponding MODIS products have been published."
        )

    image_list = collection.toList(count)
    images: List[Tuple[str, ee.Image]] = []
    for index in range(count):
        image = ee.Image(image_list.get(index))
        date_str = ee.Date(image.get("system:time_start")).format("YYYYMMdd").getInfo()
        images.append((date_str, image))
    return images


# =============================================================================
# 4) Download one image
# =============================================================================
def download_image(date_str: str, image: ee.Image, region: ee.Geometry) -> Path:
    out_name = f"{NAME_PREFIX}_{date_str}_MODIS500m"
    out_path = OUT_DIR / f"{out_name}.tif"

    if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] already exists: {out_path}")
        return out_path

    # clip() masks pixels outside Guangdong; region controls the downloaded extent.
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
    print(f"[SAVED] {date_str}: {out_path}")
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
    print(f"[INFO] download date range: [{START_DATE}, {END_DATE})")

    images = get_images()
    print("[INFO] total MCD15A3H images:", len(images))

    for date_str, image in images:
        print("[DOWNLOAD]", date_str)
        download_image(date_str, image, region)

    print("[DONE] LAI/FPAR download completed.")


if __name__ == "__main__":
    main()