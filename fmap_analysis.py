"""Analysis layer for FMAP-AI.

Consumes outputs written by fmap_download.run_download_pipeline(job_dir, ...)

Returns a JSON-serializable dict intended to be returned to a .NET frontend.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from fmap_download import NLCD_LEGEND


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exists(job_dir: str, filename: str) -> Optional[str]:
    p = os.path.join(job_dir, filename)
    return p if os.path.exists(p) else None


def _read_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _sample_tif_point(tif_path: str, lon: float, lat: float) -> Optional[float]:
    try:
        from fmap_download import sample_raster_point
        return float(sample_raster_point(tif_path, lon, lat))
    except Exception:
        return None


def _raster_stats(tif_path: str, mask_path: Optional[str] = None, resample_mask: bool = True) -> Dict[str, Any]:
    """Compute basic stats. If mask_path provided, uses mask==1."""
    out: Dict[str, Any] = {}
    if not tif_path or not os.path.exists(tif_path):
        return out

    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None:
            valid = valid & (arr != nodata)

        if mask_path and os.path.exists(mask_path):
            with rasterio.open(mask_path) as msrc:
                m = msrc.read(1)
                # resample mask to src grid if needed
                if resample_mask and (msrc.width != src.width or msrc.height != src.height or msrc.crs != src.crs or msrc.transform != src.transform):
                    dst_m = np.zeros((src.height, src.width), dtype=np.uint8)
                    reproject(
                        source=m.astype(np.uint8),
                        destination=dst_m,
                        src_transform=msrc.transform,
                        src_crs=msrc.crs,
                        dst_transform=src.transform,
                        dst_crs=src.crs,
                        resampling=Resampling.nearest,
                        src_nodata=msrc.nodata,
                        dst_nodata=0,
                    )
                    m = dst_m
                valid = valid & (m == 1)

        v = arr[valid]
        if v.size == 0:
            return out

        out["mean"] = float(np.nanmean(v))
        out["min"] = float(np.nanmin(v))
        out["max"] = float(np.nanmax(v))
        out["p10"] = float(np.nanpercentile(v, 10))
        out["p50"] = float(np.nanpercentile(v, 50))
        out["p90"] = float(np.nanpercentile(v, 90))
        out["count"] = int(v.size)
        return out


def build_manifest(job_dir: str) -> Dict[str, Any]:
    files = []
    total = 0
    for fn in sorted(os.listdir(job_dir)):
        p = os.path.join(job_dir, fn)
        if not os.path.isfile(p):
            continue
        sz = os.path.getsize(p)
        total += sz
        files.append(
            {
                "name": fn,
                "bytes": int(sz),
                "modified_utc": datetime.fromtimestamp(os.path.getmtime(p), tz=timezone.utc).isoformat(),
            }
        )
    return {"job_dir": job_dir, "total_bytes": int(total), "files": files}


def run_analysis(job_dir: str, request: Dict[str, Any], download_meta: Dict[str, Any]) -> Dict[str, Any]:
    # Paths
    ndvi_tif = _exists(job_dir, "ndvi_bbox.tif")
    ndmi_tif = _exists(job_dir, "ndmi_bbox.tif")
    nbr_tif  = _exists(job_dir, "nbr_bbox.tif")

    canopy_tif = _exists(job_dir, "nlcd_canopy_bbox.tif")
    land_tif   = _exists(job_dir, "nlcd_landcover_bbox.tif")
    forest_mask_tif = _exists(job_dir, "forest_mask_bbox.tif")

    clim_csv = _exists(job_dir, "climate_point_clean.csv")
    spi_csv  = _exists(job_dir, "spi30_point.csv")

    wfigs_gj = _exists(job_dir, "wfigs_perimeters_bbox.geojson")
    burned_csv = _exists(job_dir, "burned_area_from_perimeters.csv")
    mtbs_tif = _exists(job_dir, "mtbs_burn_severity_bbox.tif")

    agb_mg = _exists(job_dir, "fia_agb_MgHa_bbox.tif")
    agb_lb = _exists(job_dir, "fia_agb_lb_ac_bbox.tif")
    agc_mg = _exists(job_dir, "fia_agc_MgHa_bbox.tif")
    agc_lb = _exists(job_dir, "fia_agc_lb_ac_bbox.tif")
    carbon_loss = _exists(job_dir, "carbon_loss_proxy_MgHa_bbox.tif")

    # Request basics
    pt_lon = float(request.get("pt_lon"))
    pt_lat = float(request.get("pt_lat"))
    bbox = (float(request.get("minlon")), float(request.get("minlat")), float(request.get("maxlon")), float(request.get("maxlat")))

    analysis: Dict[str, Any] = {
        "generated_at": _utc_now(),
        "request": request,
        "download_meta": download_meta,
        "point": {"lon": pt_lon, "lat": pt_lat},
        "bbox": {"minlon": bbox[0], "minlat": bbox[1], "maxlon": bbox[2], "maxlat": bbox[3]},
    }

    # Landcover / canopy summary
    lc_code = download_meta.get("nlcd_landcover_code_point")
    lc_label = download_meta.get("nlcd_landcover_label_point")
    canopy_pt = download_meta.get("canopy_point_pct")
    forest_pt = download_meta.get("forest_mask_point")
    forest_frac = download_meta.get("forest_fraction_bbox")

    is_forest_point = bool(int(forest_pt) == 1) if forest_pt is not None else None

    analysis["landcover"] = {
        "nlcd_code_point": int(lc_code) if lc_code is not None else None,
        "nlcd_label_point": lc_label if lc_label is not None else (NLCD_LEGEND.get(int(lc_code), "Unknown") if lc_code is not None else None),
        "canopy_pct_point": float(canopy_pt) if canopy_pt is not None else None,
        "forest_mask_point": int(forest_pt) if forest_pt is not None else None,
        "forest_fraction_bbox": float(forest_frac) if forest_frac is not None else None,
        "is_forest_point": is_forest_point,
        "note": ("Point is not NLCD forest (41/42/43); forest-only metrics are computed over nearby forest pixels within bbox." if is_forest_point is False else None),
    }

    # Vegetation indices summary
    analysis["vegetation"] = {
        "landsat_item_id": download_meta.get("landsat_item_id"),
        "landsat_cloud_cover": download_meta.get("landsat_cloud_cover"),
        "ndvi": {"point": download_meta.get("ndvi_point"), "bbox_mean": download_meta.get("ndvi_mean_bbox")},
        "ndmi": {"point": download_meta.get("ndmi_point"), "bbox_mean": download_meta.get("ndmi_mean_bbox")},
        "nbr":  {"point": download_meta.get("nbr_point"),  "bbox_mean": download_meta.get("nbr_mean_bbox")},
        "ndvi_forest_mean_bbox": download_meta.get("ndvi_forest_mean_bbox"),
    }

    # Also compute extra stats if files are present (optional)
    if ndvi_tif:
        analysis["vegetation"]["ndvi_stats_bbox"] = _raster_stats(ndvi_tif)
        if forest_mask_tif:
            analysis["vegetation"]["ndvi_stats_forest"] = _raster_stats(ndvi_tif, mask_path=forest_mask_tif)

    # Climate summary
    dfc = _read_csv(clim_csv)
    climate = {}
    if dfc is not None and not dfc.empty:
        if "time" in dfc.columns:
            dfc["time"] = pd.to_datetime(dfc["time"], errors="coerce")
        if "pr_mm" in dfc.columns:
            climate["pr_total_mm"] = float(np.nansum(dfc["pr_mm"].values.astype("float32")))
            climate["pr_mean_mm_per_day"] = float(np.nanmean(dfc["pr_mm"].values.astype("float32")))
        if "tmax_C" in dfc.columns:
            climate["tmax_mean_C"] = float(np.nanmean(dfc["tmax_C"].values.astype("float32")))
        if "tmin_C" in dfc.columns:
            climate["tmin_mean_C"] = float(np.nanmean(dfc["tmin_C"].values.astype("float32")))
        if "tmean_C" in dfc.columns:
            climate["tmean_mean_C"] = float(np.nanmean(dfc["tmean_C"].values.astype("float32")))
        if "vpd" in dfc.columns:
            climate["vpd_mean"] = float(np.nanmean(dfc["vpd"].values.astype("float32")))
        climate["n_days"] = int(len(dfc))
    analysis["climate"] = climate

    # SPI summary
    dfs = _read_csv(spi_csv)
    spi = {}
    if dfs is not None and not dfs.empty and "spi30" in dfs.columns:
        v = dfs["spi30"].values.astype("float32")
        spi["mean"] = float(np.nanmean(v))
        spi["min"] = float(np.nanmin(v))
        spi["max"] = float(np.nanmax(v))
        try:
            spi["last"] = float(v[~np.isnan(v)][-1])
        except Exception:
            spi["last"] = None
        spi["n"] = int(np.isfinite(v).sum())
    analysis["spi30"] = spi

    # Disturbance summary (mostly from meta)
    dist = {
        "wfigs_features_in_bbox": int(download_meta.get("wfigs_features", 0) or 0),
        "burned_area_total_km2": float(download_meta.get("burned_area_total_km2", 0.0) or 0.0),
        "mtbs_layer_name": download_meta.get("mtbs_layer_name"),
    }
    if burned_csv and os.path.getsize(burned_csv) > 5:
        try:
            dist["burned_area_table_rows"] = int(len(pd.read_csv(burned_csv)))
        except Exception:
            pass
    analysis["disturbance"] = dist

    # Biomass / carbon
    biomass = {
        "agb": {
            "point_lb_ac": download_meta.get("agb_point_lb_ac") if download_meta.get("agb_point_lb_ac") is not None else _sample_tif_point(agb_lb, pt_lon, pt_lat) if agb_lb else None,
            "point_MgHa":  download_meta.get("agb_point_MgHa")  if download_meta.get("agb_point_MgHa")  is not None else _sample_tif_point(agb_mg, pt_lon, pt_lat) if agb_mg else None,
            "layer_name": download_meta.get("agb_layer_name"),
            "unit_detected": download_meta.get("agb_unit_detected"),
        },
        "agc": {
            "point_lb_ac": download_meta.get("agc_point_lb_ac") if download_meta.get("agc_point_lb_ac") is not None else _sample_tif_point(agc_lb, pt_lon, pt_lat) if agc_lb else None,
            "point_MgHa":  download_meta.get("agc_point_MgHa")  if download_meta.get("agc_point_MgHa")  is not None else _sample_tif_point(agc_mg, pt_lon, pt_lat) if agc_mg else None,
            "layer_name": download_meta.get("agc_layer_name"),
            "unit_detected": download_meta.get("agc_unit_detected"),
        },
        "notes": [],
    }

    # Forest-only means over bbox (if forest mask exists)
    if forest_mask_tif:
        if agb_mg:
            biomass["agb"]["forest_stats_MgHa"] = _raster_stats(agb_mg, mask_path=forest_mask_tif)
        if agc_mg:
            biomass["agc"]["forest_stats_MgHa"] = _raster_stats(agc_mg, mask_path=forest_mask_tif)

    if is_forest_point is False:
        biomass["notes"].append("Point is not classified as forest by NLCD; AGB/AGC point values may not be meaningful (products are for forested land).")

    analysis["biomass_carbon"] = biomass

    # Carbon loss proxy (placeholder)
    if carbon_loss:
        analysis["carbon_loss_proxy"] = _raster_stats(carbon_loss)

    # Save analysis json into job_dir for auditing/download
    out_json = os.path.join(job_dir, "analysis_result.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
    except Exception:
        pass

    return analysis
