"""Download pipeline for FMAP-AI (adapted from Colab V12).

This module is intentionally "plain Python" (no notebooks, no shell installs).
It writes outputs to a per-job directory and returns a small metadata dict.

NOTE:
- Raster downloads can be heavy. Keep bbox small for production calls.
- Render filesystem is ephemeral; use include_zip if you want to fetch outputs.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds
from rasterio.warp import reproject
from pyproj import Transformer, Geod
import pystac_client
import planetary_computer
from scipy.stats import gamma, norm

# -------------------------
# Constants / legends
# -------------------------

NLCD_LEGEND = {
    11: "Open Water", 12: "Perennial Ice/Snow", 21: "Developed, Open Space", 22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity", 24: "Developed, High Intensity", 31: "Barren",
    41: "Deciduous Forest", 42: "Evergreen Forest", 43: "Mixed Forest",
    52: "Shrub/Scrub", 71: "Grassland/Herbaceous", 81: "Pasture/Hay", 82: "Cultivated Crops",
    90: "Woody Wetlands", 95: "Emergent Herbaceous Wetlands"
}

WCS_CANOPY  = "https://dmsdata.cr.usgs.gov/geoserver/mrlc_NLCD-Tree-Canopy-Native_conus_year_data/wcs"
WCS_LANDCOV = "https://dmsdata.cr.usgs.gov/geoserver/mrlc_Land-Cover-Native_conus_year_data/wcs"

WFIGS_PERIMETERS_LAYER = "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters/FeatureServer/0"
MTBS_CONUS_IMG = "https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation/USFS_EDW_MTBS_CONUS/ImageServer"

BIOMASS_IMG = "https://imagery.geoplatform.gov/iipp/rest/services/Ecosystems/USFS_EDW_FIA_AboveGroundForestBiomass/ImageServer"
CARBON_IMG  = "https://imagery.geoplatform.gov/iipp/rest/services/Ecosystems/USFS_EDW_FIA_AboveGroundForestCarbon/ImageServer"
FOREST_ATLAS_IMG = "https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_FIA_ForestAtlas_TreeSpecies_109/ImageServer"

LB_PER_ACRE_TO_MG_PER_HA = 0.00112085116  # pounds/acre -> Mg/ha
_geod = Geod(ellps="WGS84")


# -------------------------
# Small helpers
# -------------------------

def _ncss_time(t: str) -> str:
    return t if "T" in t else f"{t}T00:00:00Z"

def safe_get(url: str, params=None, timeout: int = 180) -> requests.Response:
    headers = {"User-Agent": "FMAP-AI/0.1 (Render API)"}
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {r.url}: {(r.text or '')[:200]}")
    return r

def pick_asset_key(item, candidates: List[str]) -> str:
    keys = set(item.assets.keys())
    for c in candidates:
        if c in keys:
            return c
    raise KeyError(f"None of these assets exist: {candidates}. Available: {sorted(list(keys))[:60]} ...")

def apply_stac_scale_offset(item, asset_key: str, arr: np.ndarray) -> np.ndarray:
    asset = item.assets[asset_key]
    rb = asset.extra_fields.get("raster:bands")
    if rb and isinstance(rb, list) and len(rb) > 0:
        scale = rb[0].get("scale", 1.0)
        offset = rb[0].get("offset", 0.0)
        return arr.astype("float32") * float(scale) + float(offset)
    return arr.astype("float32")

def read_cog_subset(cog_url: str, bbox_lonlat=None, point_lonlat=None, band: int = 1):
    with rasterio.open(cog_url) as src:
        crs = src.crs
        to_src = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

        if point_lonlat is not None:
            lon, lat = point_lonlat
            x, y = to_src.transform(lon, lat)
            row, col = src.index(x, y)
            window = rasterio.windows.Window(col_off=col, row_off=row, width=1, height=1)
            arr = src.read(band, window=window)
            return float(arr[0, 0]), src.profile

        if bbox_lonlat is None:
            raise ValueError("Provide bbox_lonlat or point_lonlat")

        minlon, minlat, maxlon, maxlat = bbox_lonlat
        xmin, ymin = to_src.transform(minlon, minlat)
        xmax, ymax = to_src.transform(maxlon, maxlat)

        win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        win = win.round_offsets().round_lengths()
        arr = src.read(band, window=win)
        w_transform = src.window_transform(win)

        profile = src.profile.copy()
        profile.update({"height": arr.shape[0], "width": arr.shape[1], "transform": w_transform})
        return arr, profile, w_transform

def write_geotiff(path: str, arr: np.ndarray, profile: dict, dtype="float32", nodata=None) -> str:
    prof = profile.copy()
    prof.update(count=1, dtype=dtype, nodata=nodata, tiled=False, compress="deflate")
    for k in ["blockxsize", "blockysize", "interleave"]:
        prof.pop(k, None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return path

def sample_raster_point(path: str, lon: float, lat: float, band: int = 1) -> float:
    with rasterio.open(path) as src:
        to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = to_src.transform(lon, lat)
        row, col = src.index(x, y)
        return float(src.read(band, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])

def resample_to_profile(src_path: str, dst_profile: dict, resampling=Resampling.nearest, dst_dtype="uint8", dst_nodata=0) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_profile = src.profile

        dst_arr = np.full((dst_profile["height"], dst_profile["width"]), dst_nodata, dtype=dst_dtype)

        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            resampling=resampling,
            src_nodata=src_profile.get("nodata", None),
            dst_nodata=dst_nodata,
        )
        return dst_arr

# ---- gridMET packed value decoding (scale_factor/add_offset) ----
# The gridMET THREDDS/NCSS CSV responses often return packed integer values.
# We decode them using the dataset's CF attributes (scale_factor, add_offset).
_GRIDMET_SCALE_CACHE: Dict[tuple, tuple] = {}

# Safe defaults (as of Feb 2026). Used if metadata lookup fails.
# Tuple: (scale_factor, add_offset)
_GRIDMET_DEFAULTS: Dict[tuple, tuple] = {
    ("pr",   "precipitation_amount"): (0.1, 0.0),   # mm
    ("tmmx", "daily_maximum_temperature"): (0.1, 220.0),  # K
    ("tmmn", "daily_minimum_temperature"): (0.1, 210.0),  # K
    ("vpd",  "daily_mean_vapor_pressure_deficit"): (0.01, 0.0),  # kPa
}

def gridmet_scale_offset(dataset_id: str, varname: str) -> tuple:
    key = (dataset_id, varname)
    if key in _GRIDMET_SCALE_CACHE:
        return _GRIDMET_SCALE_CACHE[key]

    # Try to read attributes from the dataset's OPeNDAP HTML page (fast + no NetCDF client needed)
    url = f"https://tds-proxy.nkn.uidaho.edu/thredds/dodsC/agg_met_{dataset_id}_1979_CurrentYear_CONUS.nc.html"
    try:
        r = safe_get(url, timeout=30)
        txt = r.text or ""
        # Variable line typically contains: "... scale_factor: X add_offset: Y ..."
        pat = re.compile(rf"{re.escape(varname)}.*?scale_factor:\s*([0-9eE.+\-]+).*?add_offset:\s*([0-9eE.+\-]+)", re.S)
        m = pat.search(txt)
        if m:
            sf = float(m.group(1))
            off = float(m.group(2))
            _GRIDMET_SCALE_CACHE[key] = (sf, off)
            return sf, off
    except Exception:
        pass

    if key in _GRIDMET_DEFAULTS:
        _GRIDMET_SCALE_CACHE[key] = _GRIDMET_DEFAULTS[key]
        return _GRIDMET_DEFAULTS[key]

    # Fallback (no scaling)
    _GRIDMET_SCALE_CACHE[key] = (1.0, 0.0)
    return (1.0, 0.0)

def gridmet_decode(series: pd.Series, dataset_id: str, varname: Optional[str]) -> pd.Series:
    s = series.astype("float32")
    if not varname:
        return s
    sf, off = gridmet_scale_offset(dataset_id, varname)
    return s * float(sf) + float(off)

def gridmet_decode_temp_C(series: pd.Series, dataset_id: str, varname: Optional[str], kelvin_to_C=True) -> pd.Series:
    k = gridmet_decode(series, dataset_id, varname)
    return (k - 273.15) if kelvin_to_C else k

# -------------------------
# SPI helpers
# -------------------------

def _spi_from_gamma(x_values, a, scale, p0):
    g = gamma.cdf(x_values, a, loc=0, scale=scale)
    H = p0 + (1.0 - p0) * g
    H = np.clip(H, 1e-6, 1.0 - 1e-6)
    return norm.ppf(H)

def spi_gamma_monthly(series: pd.Series, window=30, min_samples_per_month=60, fallback_to_pooled=True) -> pd.Series:
    x = series.rolling(window=window, min_periods=window).sum()
    spi = pd.Series(index=series.index, data=np.nan, dtype="float32")

    pooled_params = None
    if fallback_to_pooled:
        x_valid = x.dropna()
        if len(x_valid) >= 12 * min_samples_per_month:
            p0p = float((x_valid.values <= 0).mean())
            x_fitp = x_valid[x_valid > 0].values
            if len(x_fitp) >= max(120, 2 * min_samples_per_month):
                ap, locp, scalep = gamma.fit(x_fitp, floc=0)
                pooled_params = (ap, scalep, p0p)

    for m in range(1, 13):
        sel = (x.index.month == m) & x.notna()
        x_m = x[sel]
        if x_m.empty:
            continue

        p0 = float((x_m.values <= 0).mean())
        x_fit = x_m[x_m > 0].values

        if len(x_m) >= min_samples_per_month and len(x_fit) >= max(30, min_samples_per_month // 3):
            a, loc, scale = gamma.fit(x_fit, floc=0)
            spi.loc[sel] = _spi_from_gamma(x_m.values, a, scale, p0).astype("float32")
        elif pooled_params is not None:
            ap, scalep, p0p = pooled_params
            spi.loc[sel] = _spi_from_gamma(x_m.values, ap, scalep, p0p).astype("float32")

    return spi

# -------------------------
# ArcGIS ImageServer helpers
# -------------------------

def arcgis_imageserver_list_rasters(img_url: str) -> List[dict]:
    q_url = img_url.rstrip("/") + "/query"
    params = {
        "where": "1=1",
        "outFields": "OBJECTID,name,Name",
        "returnGeometry": "false",
        "f": "pjson",
        "resultRecordCount": "2000",
    }
    r = safe_get(q_url, params=params, timeout=180).json()
    feats = r.get("features", []) or []
    out = []
    for f in feats:
        a = f.get("attributes", {}) or {}
        oid = a.get("OBJECTID", a.get("objectid", a.get("ObjectId")))
        nm = a.get("name", a.get("Name", ""))
        out.append({"objectid": oid, "name": nm})
    return out

def arcgis_export_geotiff(img_url: str, bbox_lonlat: Tuple[float,float,float,float], out_tif: str, size=(900, 900), raster_pick_regex: Optional[str]=None):
    img_url = img_url.rstrip("/")
    w, h = size
    minlon, minlat, maxlon, maxlat = bbox_lonlat

    mosaicRule = None
    picked_name = None
    if raster_pick_regex:
        try:
            rasters = arcgis_imageserver_list_rasters(img_url)
            pat = re.compile(raster_pick_regex, re.IGNORECASE)
            hit = next((r for r in rasters if r["objectid"] is not None and pat.search(str(r["name"] or ""))), None)
            if hit:
                mosaicRule = {"where": f"OBJECTID={hit['objectid']}"}
                picked_name = hit.get("name")
        except Exception:
            mosaicRule = None

    params = {
        "f": "pjson",
        "bbox": f"{minlon},{minlat},{maxlon},{maxlat}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{w},{h}",
        "format": "tiff",
        "pixelType": "UNKNOWN",
        "noDataInterpretation": "esriNoDataMatchAny",
    }
    if mosaicRule is not None:
        params["mosaicRule"] = json.dumps(mosaicRule)

    meta = safe_get(img_url + "/exportImage", params=params, timeout=300).json()
    href = meta.get("href")
    ext = meta.get("extent") or {}
    if not href or not ext:
        raise RuntimeError("exportImage JSON missing href/extent")

    raw = safe_get(href, timeout=300).content
    raw_path = out_tif.replace(".tif", "_raw.tif")
    with open(raw_path, "wb") as f:
        f.write(raw)

    xmin = ext.get("xmin"); ymin = ext.get("ymin"); xmax = ext.get("xmax"); ymax = ext.get("ymax")
    if None in [xmin, ymin, xmax, ymax]:
        xmin, ymin, xmax, ymax = minlon, minlat, maxlon, maxlat

    with rasterio.open(raw_path) as src:
        arr = src.read(1)

    transform = transform_from_bounds(xmin, ymin, xmax, ymax, arr.shape[1], arr.shape[0])
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": arr.dtype,
        "crs": "EPSG:4326",
        "transform": transform,
        "tiled": False,
        "compress": "deflate",
        "nodata": None,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)

    return out_tif, picked_name

# -------------------------
# ArcGIS FeatureServer bbox -> GeoJSON
# -------------------------

def arcgis_feature_query_geojson(layer_url: str, bbox_lonlat, out_geojson: str, where="1=1", out_sr=4326, timeout=300):
    minlon, minlat, maxlon, maxlat = bbox_lonlat
    q_url = layer_url.rstrip("/") + "/query"
    params = {
        "where": where,
        "geometry": f"{minlon},{minlat},{maxlon},{maxlat}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": str(out_sr),
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": str(out_sr),
        "resultRecordCount": "2000",
    }
    r = safe_get(q_url, params=params, timeout=timeout)
    gj = r.json()
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(gj, f)
    n = len(gj.get("features", [])) if isinstance(gj, dict) else 0
    return out_geojson, n

# -------------------------
# Burned area from GeoJSON (no shapely)
# -------------------------

def _poly_area_km2(coords) -> float:
    area_m2_total = 0.0
    for ring in coords:
        if not ring or len(ring) < 3:
            continue
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        area_m2, _ = _geod.polygon_area_perimeter(lons, lats)
        area_m2_total += abs(area_m2)
    return area_m2_total / 1e6

def burned_area_from_geojson_km2(geojson_path: str, out_csv: Optional[str]=None) -> Tuple[float, int]:
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    feats = gj.get("features", []) if isinstance(gj, dict) else []
    total_km2 = 0.0
    rows = []
    for feat in feats:
        geom = (feat or {}).get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        props = (feat or {}).get("properties") or {}

        km2 = 0.0
        if gtype == "Polygon":
            km2 = _poly_area_km2(coords)
        elif gtype == "MultiPolygon":
            for poly in coords:
                km2 += _poly_area_km2(poly)

        total_km2 += km2
        rows.append({
            "id": props.get("OBJECTID", props.get("objectid", props.get("id"))),
            "name": props.get("IncidentName", props.get("incidentname", props.get("name"))),
            "area_km2": km2
        })

    if out_csv:
        df = pd.DataFrame(rows, columns=["id","name","area_km2"])
        df.to_csv(out_csv, index=False)
    return float(total_km2), len(rows)

# -------------------------
# MRLC WCS helpers
# -------------------------

def wcs_list_coverages(wcs_url: str, version="1.0.0") -> List[str]:
    params = {"service": "WCS", "version": version, "request": "GetCapabilities"}
    r = safe_get(wcs_url, params=params, timeout=180)
    root = ET.fromstring(r.content)

    paths = [
        ".//{*}CoverageOfferingBrief/{*}name",
        ".//{*}CoverageSummary/{*}Identifier",
        ".//{*}Identifier",
        ".//{*}name",
    ]

    names: List[str] = []
    for p in paths:
        for el in root.findall(p):
            if el.text and el.text.strip():
                names.append(el.text.strip())

    seen = set()
    out = []
    for n in names:
        if n.upper() == "WCS":
            continue
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out

def pick_latest_year_coverage(coverage_names: List[str], prefer_keywords: Optional[List[str]] = None) -> Optional[str]:
    prefer_keywords = prefer_keywords or []

    def year_of(s: str):
        m = re.search(r"(19|20)\d{2}", s)
        return int(m.group()) if m else None

    cands = coverage_names[:]
    if prefer_keywords:
        c2 = [c for c in cands if any(k.lower() in c.lower() for k in prefer_keywords)]
        if c2:
            cands = c2

    scored = []
    for c in cands:
        y = year_of(c)
        scored.append((y if y is not None else -1, c))
    scored.sort(reverse=True)
    return scored[0][1] if scored else None

def wcs_getcoverage_geotiff(wcs_url: str, coverage_name: str, bbox_lonlat, out_tif: str, width=900, height=900, version="1.0.0") -> str:
    minlon, minlat, maxlon, maxlat = bbox_lonlat
    params = {
        "service": "WCS",
        "version": version,
        "request": "GetCoverage",
        "coverage": coverage_name,
        "crs": "EPSG:4326",
        "bbox": f"{minlon},{minlat},{maxlon},{maxlat}",
        "format": "GeoTIFF",
        "width": str(width),
        "height": str(height),
    }
    r = safe_get(wcs_url, params=params, timeout=300)
    with open(out_tif, "wb") as f:
        f.write(r.content)
    return out_tif

# -------------------------
# gridMET (THREDDS NCSS)
# -------------------------

GRIDMET_DATASETS = {
    "pr":   {"dataset_id": "pr",   "vars": ["precipitation_amount", "pr"]},
    "tmmx": {"dataset_id": "tmmx", "vars": ["daily_maximum_temperature", "tmmx"]},
    "tmmn": {"dataset_id": "tmmn", "vars": ["daily_minimum_temperature", "tmmn"]},
    "vpd":  {"dataset_id": "vpd",  "vars": ["daily_mean_vapor_pressure_deficit", "mean_vapor_pressure_deficit", "vapor_pressure_deficit", "vpd"]},
}

def gridmet_ncss_url(dataset_id: str) -> str:
    return f"https://tds-proxy.nkn.uidaho.edu/thredds/ncss/grid/agg_met_{dataset_id}_1979_CurrentYear_CONUS.nc"

def ncss_point_csv(dataset_id: str, var_candidates: List[str], lon: float, lat: float, start: str, end: str, out_csv: str) -> Tuple[str, pd.DataFrame, str]:
    url = gridmet_ncss_url(dataset_id)
    last = None
    for varname in var_candidates:
        params = {
            "var": varname,
            "latitude": lat,
            "longitude": lon,
            "time_start": _ncss_time(start),
            "time_end": _ncss_time(end),
            "accept": "csv",
        }
        r = requests.get(url, params=params, timeout=180)
        last = r
        if r.status_code == 200 and r.content:
            with open(out_csv, "wb") as f:
                f.write(r.content)
            return out_csv, pd.read_csv(out_csv), varname

    raise RuntimeError(f"No variable name worked for NCSS point request. Last: {getattr(last,'url',None)}")

def ncss_bbox_netcdf(dataset_id: str, var_candidates: List[str], bbox_lonlat, start: str, end: str, out_nc: str) -> Tuple[str, str]:
    minlon, minlat, maxlon, maxlat = bbox_lonlat
    url = gridmet_ncss_url(dataset_id)
    last = None
    for varname in var_candidates:
        params = {
            "var": varname,
            "north": maxlat,
            "south": minlat,
            "east": maxlon,
            "west": minlon,
            "time_start": _ncss_time(start),
            "time_end": _ncss_time(end),
            "accept": "netcdf",
        }
        r = requests.get(url, params=params, timeout=300)
        last = r
        if r.status_code == 200 and r.content:
            with open(out_nc, "wb") as f:
                f.write(r.content)
            return out_nc, varname

    raise RuntimeError(f"No variable name worked for NCSS bbox request. Last: {getattr(last,'url',None)}")


# -------------------------
# FIA unit detection/conversion helpers
# -------------------------

def detect_unit_from_name_or_stats(layer_name: Optional[str], arr: np.ndarray) -> str:
    name = (layer_name or "").lower()

    if any(k in name for k in ["mg_ha", "mg/ha", "mg ha", "megagram", "ton_ha", "t_ha", "ton/ha", "t/ha"]):
        return "Mg/ha"
    if any(k in name for k in ["lb", "lbs", "lb_ac", "lb/ac", "pound", "lb per acre", "pounds"]):
        return "lb/ac"

    v = arr.astype("float32")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "unknown"

    p99 = float(np.nanpercentile(v, 99))
    med = float(np.nanmedian(v))

    if p99 > 1200 or med > 600:
        return "lb/ac"
    return "Mg/ha"

def write_unit_variants(raw_tif: str, layer_name: Optional[str], out_lb_ac_tif: str, out_mg_ha_tif: str) -> str:
    with rasterio.open(raw_tif) as src:
        arr = src.read(1).astype("float32")
        prof = src.profile.copy()

    unit = detect_unit_from_name_or_stats(layer_name, arr)

    if unit == "lb/ac":
        arr_lb = arr
        arr_mg = arr * LB_PER_ACRE_TO_MG_PER_HA
    elif unit == "Mg/ha":
        arr_mg = arr
        arr_lb = arr / LB_PER_ACRE_TO_MG_PER_HA
    else:
        return unit

    write_geotiff(out_lb_ac_tif, arr_lb, prof, dtype="float32")
    write_geotiff(out_mg_ha_tif, arr_mg, prof, dtype="float32")
    return unit


# -------------------------
# STAC client (cached)
# -------------------------

@lru_cache(maxsize=1)
def _stac_client():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


# -------------------------
# Main pipeline
# -------------------------

def run_download_pipeline(
    job_dir: str,
    pt_lon: float,
    pt_lat: float,
    bbox_lonlat: Tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    spi_start: str,
    cloud_cover_lt: float = 30.0,
    size_px: int = 900,
) -> Dict[str, Union[str, float, int, None]]:
    """Run the full download pipeline and return a metadata dict."""

    os.makedirs(job_dir, exist_ok=True)

    meta: Dict[str, Union[str, float, int, None]] = {
        "pt_lon": pt_lon,
        "pt_lat": pt_lat,
        "bbox": bbox_lonlat,
        "date_start": date_start,
        "date_end": date_end,
        "spi_start": spi_start,
    }

    # -------------------------
    # 1) Landsat vegetation indices
    # -------------------------
    catalog = _stac_client()
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=list(bbox_lonlat),
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_lt}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No Landsat scenes found. Try wider date range or larger bbox.")
    item = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 999))[0]
    meta["landsat_item_id"] = item.id
    meta["landsat_cloud_cover"] = float(item.properties.get("eo:cloud_cover", np.nan))

    red_key   = pick_asset_key(item, ["red", "SR_B4", "B04"])
    nir_key   = pick_asset_key(item, ["nir08", "SR_B5", "B08"])
    swir1_key = pick_asset_key(item, ["swir16", "SR_B6", "B11"])
    swir2_key = pick_asset_key(item, ["swir22", "SR_B7", "B12"])

    red, landsat_prof, _   = read_cog_subset(item.assets[red_key].href,   bbox_lonlat=bbox_lonlat)
    nir, _, _              = read_cog_subset(item.assets[nir_key].href,   bbox_lonlat=bbox_lonlat)
    swir1, _, _            = read_cog_subset(item.assets[swir1_key].href, bbox_lonlat=bbox_lonlat)
    swir2, _, _            = read_cog_subset(item.assets[swir2_key].href, bbox_lonlat=bbox_lonlat)

    red_f   = apply_stac_scale_offset(item, red_key, red)
    nir_f   = apply_stac_scale_offset(item, nir_key, nir)
    swir1_f = apply_stac_scale_offset(item, swir1_key, swir1)
    swir2_f = apply_stac_scale_offset(item, swir2_key, swir2)

    ndvi = (nir_f - red_f) / (nir_f + red_f + 1e-6)
    ndmi = (nir_f - swir1_f) / (nir_f + swir1_f + 1e-6)
    nbr  = (nir_f - swir2_f) / (nir_f + swir2_f + 1e-6)

    ndvi_path = os.path.join(job_dir, "ndvi_bbox.tif")
    ndmi_path = os.path.join(job_dir, "ndmi_bbox.tif")
    nbr_path  = os.path.join(job_dir, "nbr_bbox.tif")
    write_geotiff(ndvi_path, ndvi, landsat_prof, dtype="float32")
    write_geotiff(ndmi_path, ndmi, landsat_prof, dtype="float32")
    write_geotiff(nbr_path,  nbr,  landsat_prof, dtype="float32")

    meta["ndvi_mean_bbox"] = float(np.nanmean(ndvi))
    meta["ndmi_mean_bbox"] = float(np.nanmean(ndmi))
    meta["nbr_mean_bbox"]  = float(np.nanmean(nbr))
    meta["ndvi_point"] = float(sample_raster_point(ndvi_path, pt_lon, pt_lat))
    meta["ndmi_point"] = float(sample_raster_point(ndmi_path, pt_lon, pt_lat))
    meta["nbr_point"]  = float(sample_raster_point(nbr_path,  pt_lon, pt_lat))

    # -------------------------
    # 2) NLCD canopy + landcover (forest mask + proxy group)
    # -------------------------
    canopy_covs = wcs_list_coverages(WCS_CANOPY, version="1.0.0")
    canopy_cov  = pick_latest_year_coverage(canopy_covs, prefer_keywords=["canopy", "tree"])
    if not canopy_cov:
        raise RuntimeError("Could not find NLCD canopy coverage in WCS capabilities.")
    canopy_tif  = os.path.join(job_dir, "nlcd_canopy_bbox.tif")
    wcs_getcoverage_geotiff(WCS_CANOPY, canopy_cov, bbox_lonlat, canopy_tif, width=size_px, height=size_px)

    land_tif = os.path.join(job_dir, "nlcd_landcover_bbox.tif")
    forest_mask_tif = os.path.join(job_dir, "forest_mask_bbox.tif")
    forest_type_group_tif = os.path.join(job_dir, "forest_type_group_nlcd_bbox.tif")
    canopy_class_tif = os.path.join(job_dir, "canopy_class_bbox.tif")

    forest_type_group_exists = False

    try:
        land_covs = wcs_list_coverages(WCS_LANDCOV, version="1.0.0")
        land_cov  = pick_latest_year_coverage(land_covs, prefer_keywords=["land", "cover"])
        if not land_cov:
            raise RuntimeError("Could not find NLCD landcover coverage in WCS capabilities.")
        wcs_getcoverage_geotiff(WCS_LANDCOV, land_cov, bbox_lonlat, land_tif, width=size_px, height=size_px)

        forest_classes = {41, 42, 43}
        with rasterio.open(land_tif) as src:
            lc = src.read(1).astype("int32")

            mask = np.isin(lc, list(forest_classes)).astype("uint8")
            profile_m = src.profile.copy()
            profile_m.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=False)
            for k in ["blockxsize","blockysize","interleave"]:
                profile_m.pop(k, None)
            with rasterio.open(forest_mask_tif, "w", **profile_m) as dst:
                dst.write(mask, 1)

            # Forest type group proxy
            ft = np.zeros_like(lc, dtype="uint8")
            ft[lc == 41] = 1
            ft[lc == 42] = 2
            ft[lc == 43] = 3
            prof = src.profile.copy()
            prof.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=False)
            for k in ["blockxsize","blockysize","interleave"]:
                prof.pop(k, None)
            with rasterio.open(forest_type_group_tif, "w", **prof) as dst:
                dst.write(ft, 1)
            forest_type_group_exists = True

        meta["forest_fraction_bbox"] = float(mask.mean())

    except Exception:
        # Fallback forest mask from canopy >=10%
        with rasterio.open(canopy_tif) as src:
            c = src.read(1).astype("float32")
            mask = (c >= 10).astype("uint8")
            profile_m = src.profile.copy()
            profile_m.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=False)
            for k in ["blockxsize","blockysize","interleave"]:
                profile_m.pop(k, None)
            with rasterio.open(forest_mask_tif, "w", **profile_m) as dst:
                dst.write(mask, 1)
        meta["forest_fraction_bbox"] = float(mask.mean())
        forest_type_group_exists = False

    # canopy class 0..3
    with rasterio.open(canopy_tif) as src:
        c = src.read(1).astype("float32")
        cc = np.zeros_like(c, dtype="uint8")
        cc[(c >= 10) & (c < 40)] = 1
        cc[(c >= 40) & (c < 70)] = 2
        cc[(c >= 70)] = 3
        prof = src.profile.copy()
        prof.update(dtype="uint8", count=1, nodata=0, compress="deflate", tiled=False)
        for k in ["blockxsize","blockysize","interleave"]:
            prof.pop(k, None)
        with rasterio.open(canopy_class_tif, "w", **prof) as dst:
            dst.write(cc, 1)

    # point diagnostics
    meta["canopy_point_pct"] = float(sample_raster_point(canopy_tif, pt_lon, pt_lat))
    if os.path.exists(land_tif):
        lc_pt = int(sample_raster_point(land_tif, pt_lon, pt_lat))
        meta["nlcd_landcover_code_point"] = lc_pt
        meta["nlcd_landcover_label_point"] = NLCD_LEGEND.get(lc_pt, "Unknown")
    meta["forest_mask_point"] = int(sample_raster_point(forest_mask_tif, pt_lon, pt_lat))

    # forest-only NDVI mean (mask resampled to landsat)
    fm_landsat = resample_to_profile(forest_mask_tif, landsat_prof, resampling=Resampling.nearest, dst_dtype="uint8", dst_nodata=0)
    if float(fm_landsat.mean()) < 0.001:
        meta["ndvi_forest_mean_bbox"] = float("nan")
    else:
        meta["ndvi_forest_mean_bbox"] = float(np.nanmean(np.where(fm_landsat == 1, ndvi, np.nan)))

    # -------------------------
    # 3) gridMET point series + cleaned climate CSV
    # -------------------------
    clim_point: Dict[str, pd.DataFrame] = {}
    for k, info in GRIDMET_DATASETS.items():
        out_csv = os.path.join(job_dir, f"gridmet_{k}_point.csv")
        try:
            _, df, used_var = ncss_point_csv(info["dataset_id"], info["vars"], pt_lon, pt_lat, date_start, date_end, out_csv)
            clim_point[k] = df
            meta[f"gridmet_{k}_var"] = used_var
        except Exception:
            meta[f"gridmet_{k}_var"] = None

    # Optional bbox netcdf example (tmmx)
    try:
        out_nc = os.path.join(job_dir, "gridmet_tmmx_bbox.nc")
        _, used_var = ncss_bbox_netcdf("tmmx", GRIDMET_DATASETS["tmmx"]["vars"], bbox_lonlat, date_start, date_end, out_nc)
        meta["gridmet_tmmx_bbox_var"] = used_var
    except Exception:
        meta["gridmet_tmmx_bbox_var"] = None

    # Cleaned climate csv
    out_clim_clean = os.path.join(job_dir, "climate_point_clean.csv")
    df_pr  = clim_point.get("pr")
    df_tx  = clim_point.get("tmmx")
    df_tn  = clim_point.get("tmmn")
    df_vpd = clim_point.get("vpd")

    def _pick_numeric(df: Optional[pd.DataFrame]) -> Optional[str]:
        if df is None:
            return None
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        return num[-1] if num else None

    t = pd.to_datetime(df_pr["time"]) if df_pr is not None and "time" in df_pr.columns else None

    pr_col  = next((c for c in (df_pr.columns if df_pr is not None else []) if "precip" in c.lower()), _pick_numeric(df_pr))
    tx_col  = next((c for c in (df_tx.columns if df_tx is not None else []) if "temp" in c.lower() or "tmmx" in c.lower()), _pick_numeric(df_tx))
    tn_col  = next((c for c in (df_tn.columns if df_tn is not None else []) if "temp" in c.lower() or "tmmn" in c.lower()), _pick_numeric(df_tn))
    vpd_col = next((c for c in (df_vpd.columns if df_vpd is not None else []) if "vapor" in c.lower() or "vpd" in c.lower()), _pick_numeric(df_vpd))

    out = pd.DataFrame({"time": t})
    if pr_col:
        out["pr_mm"] = gridmet_decode(df_pr[pr_col], "pr", meta.get("gridmet_pr_var"))
    if tx_col:
        out["tmax_C"] = gridmet_decode_temp_C(df_tx[tx_col], "tmmx", meta.get("gridmet_tmmx_var"))
    if tn_col:
        out["tmin_C"] = gridmet_decode_temp_C(df_tn[tn_col], "tmmn", meta.get("gridmet_tmmn_var"))
    if "tmax_C" in out.columns and "tmin_C" in out.columns:
        out["tmean_C"] = (out["tmax_C"] + out["tmin_C"]) / 2.0
    if vpd_col:
        out["vpd"] = df_vpd[vpd_col].astype("float32")

    out.to_csv(out_clim_clean, index=False)

    # -------------------------
    # 4) SPI-30
    # -------------------------
    if "pr" in clim_point:
        spi_csv = os.path.join(job_dir, "gridmet_pr_point_for_spi.csv")
        _, dfp, used_spi_var = ncss_point_csv("pr", GRIDMET_DATASETS["pr"]["vars"], pt_lon, pt_lat, spi_start, date_end, spi_csv)
        dfp["time"] = pd.to_datetime(dfp["time"], utc=True).dt.tz_convert(None)
        dfp = dfp.sort_values("time").set_index("time")

        pr_col2 = next((c for c in dfp.columns if "precip" in c.lower()), None)
        if pr_col2 is None:
            pr_col2 = dfp.select_dtypes(include=[np.number]).columns.tolist()[-1]

        pr = gridmet_decode(dfp[pr_col2], "pr", used_spi_var)
        spi30_all = spi_gamma_monthly(pr, window=30, min_samples_per_month=60, fallback_to_pooled=True)

        start_ts = pd.to_datetime(date_start)
        end_ts   = pd.to_datetime(date_end)
        spi30_win = spi30_all.loc[start_ts:end_ts]
        out_spi = os.path.join(job_dir, "spi30_point.csv")
        pd.DataFrame({"time": spi30_win.index, "spi30": spi30_win.values}).to_csv(out_spi, index=False)

    # -------------------------
    # 5) Disturbance: WFIGS perimeters + burned area + MTBS severity
    # -------------------------
    wfigs_geojson = os.path.join(job_dir, "wfigs_perimeters_bbox.geojson")
    y0 = int(date_start[:4]); y1 = int(date_end[:4])
    where_try = f"FireYear >= {y0} AND FireYear <= {y1}"

    try:
        fire_gj_path, nfeat = arcgis_feature_query_geojson(WFIGS_PERIMETERS_LAYER, bbox_lonlat, wfigs_geojson, where=where_try)
    except Exception:
        fire_gj_path, nfeat = arcgis_feature_query_geojson(WFIGS_PERIMETERS_LAYER, bbox_lonlat, wfigs_geojson, where="1=1")
    meta["wfigs_features"] = int(nfeat)

    out_area_csv = os.path.join(job_dir, "burned_area_from_perimeters.csv")
    try:
        total_km2, nrows = burned_area_from_geojson_km2(fire_gj_path, out_csv=out_area_csv)
        meta["burned_area_total_km2"] = float(total_km2)
        meta["burned_area_features"] = int(nrows)
    except Exception:
        meta["burned_area_total_km2"] = 0.0
        meta["burned_area_features"] = 0

    mtbs_tif = os.path.join(job_dir, "mtbs_burn_severity_bbox.tif")
    try:
        _, mtbs_layer = arcgis_export_geotiff(MTBS_CONUS_IMG, bbox_lonlat, mtbs_tif, size=(size_px,size_px), raster_pick_regex=r"severity|burn|dNBR")
        meta["mtbs_layer_name"] = mtbs_layer
    except Exception:
        meta["mtbs_layer_name"] = None

    # -------------------------
    # 6) Biomass/Carbon + simple proxy outputs
    # -------------------------
    agb_raw_tif = os.path.join(job_dir, "fia_agb_raw_bbox.tif")
    agb_lb_tif  = os.path.join(job_dir, "fia_agb_lb_ac_bbox.tif")
    agb_mg_tif  = os.path.join(job_dir, "fia_agb_MgHa_bbox.tif")

    agc_raw_tif = os.path.join(job_dir, "fia_agc_raw_bbox.tif")
    agc_lb_tif  = os.path.join(job_dir, "fia_agc_lb_ac_bbox.tif")
    agc_mg_tif  = os.path.join(job_dir, "fia_agc_MgHa_bbox.tif")

    try:
        _, layer_name = arcgis_export_geotiff(BIOMASS_IMG, bbox_lonlat, agb_raw_tif, size=(size_px,size_px), raster_pick_regex=r"above|biomass|agb|dry")
        meta["agb_layer_name"] = layer_name
        meta["agb_unit_detected"] = write_unit_variants(agb_raw_tif, layer_name, agb_lb_tif, agb_mg_tif)

        meta["agb_point_lb_ac"] = float(sample_raster_point(agb_lb_tif, pt_lon, pt_lat)) if os.path.exists(agb_lb_tif) else None
        meta["agb_point_MgHa"]  = float(sample_raster_point(agb_mg_tif, pt_lon, pt_lat)) if os.path.exists(agb_mg_tif) else None
    except Exception:
        meta["agb_layer_name"] = None
        meta["agb_unit_detected"] = None
        meta["agb_point_lb_ac"] = None
        meta["agb_point_MgHa"] = None

    try:
        _, layer_name = arcgis_export_geotiff(CARBON_IMG, bbox_lonlat, agc_raw_tif, size=(size_px,size_px), raster_pick_regex=r"above|carbon|agc")
        meta["agc_layer_name"] = layer_name
        meta["agc_unit_detected"] = write_unit_variants(agc_raw_tif, layer_name, agc_lb_tif, agc_mg_tif)

        meta["agc_point_lb_ac"] = float(sample_raster_point(agc_lb_tif, pt_lon, pt_lat)) if os.path.exists(agc_lb_tif) else None
        meta["agc_point_MgHa"]  = float(sample_raster_point(agc_mg_tif, pt_lon, pt_lat)) if os.path.exists(agc_mg_tif) else None
    except Exception:
        meta["agc_layer_name"] = None
        meta["agc_unit_detected"] = None
        meta["agc_point_lb_ac"] = None
        meta["agc_point_MgHa"] = None

    # Carbon-from-biomass proxy (C ≈ 0.5 * biomass)
    carbon_from_biomass_lb_tif = os.path.join(job_dir, "carbon_from_biomass_proxy_lb_ac_bbox.tif")
    carbon_from_biomass_mg_tif = os.path.join(job_dir, "carbon_from_biomass_proxy_MgHa_bbox.tif")
    try:
        if os.path.exists(agb_lb_tif):
            with rasterio.open(agb_lb_tif) as src:
                b = src.read(1).astype("float32")
                prof = src.profile.copy()
            write_geotiff(carbon_from_biomass_lb_tif, 0.5 * b, prof, dtype="float32")
        if os.path.exists(agb_mg_tif):
            with rasterio.open(agb_mg_tif) as src:
                b = src.read(1).astype("float32")
                prof = src.profile.copy()
            write_geotiff(carbon_from_biomass_mg_tif, 0.5 * b, prof, dtype="float32")
    except Exception:
        pass

    # Forest Atlas (optional: may export default mosaic)
    try:
        forest_atlas_tif = os.path.join(job_dir, "forest_atlas_layer_bbox.tif")
        _, layer_name = arcgis_export_geotiff(FOREST_ATLAS_IMG, bbox_lonlat, forest_atlas_tif, size=(size_px,size_px), raster_pick_regex=r"forest|type|group|species")
        meta["forest_atlas_layer_name"] = layer_name
    except Exception:
        meta["forest_atlas_layer_name"] = None

    # Carbon loss proxy (placeholder): C * (MTBS != 0)
    try:
        if os.path.exists(agc_mg_tif) and os.path.exists(mtbs_tif):
            with rasterio.open(agc_mg_tif) as srcC:
                C = srcC.read(1).astype("float32")
                profC = srcC.profile.copy()
            with rasterio.open(mtbs_tif) as srcS:
                S = srcS.read(1).astype("float32")
            mask = (np.isfinite(S) & (S != 0)).astype("float32")
            out_loss = os.path.join(job_dir, "carbon_loss_proxy_MgHa_bbox.tif")
            write_geotiff(out_loss, C * mask, profC, dtype="float32")
    except Exception:
        pass

    # Save meta as json too
    meta_path = os.path.join(job_dir, "download_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
