"""FMAP-AI backend API (Render/GitHub ready)

Orchestrates:
  1) Download pipeline (adapted from Colab V12)
  2) Analysis on downloaded outputs
  3) Return JSON (and optionally a ZIP)

Render Start Command:
  uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
import time
import uuid
import shutil
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Literal, Tuple
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

import logging
import threading
import io
import faulthandler

from fmap_download import run_download_pipeline, ncss_point_csv, GRIDMET_DATASETS, spi_gamma_monthly
from fmap_analysis import run_analysis, build_manifest

APP_VERSION = "0.1.1"
JOB_ROOT = os.getenv("FMAP_JOB_ROOT", "/tmp/fmap_jobs")
MAX_JOB_AGE_SECONDS = int(os.getenv("FMAP_MAX_JOB_AGE_SECONDS", "86400"))  # 24h

# Guardrails for Render temporary storage (free instances are typically evicted around ~2GB).
# - MAX_JOB_BYTES: soft ceiling for a single job directory. If exceeded, we prune intermediates.
# - ZIP_MAX_BYTES: refuse to build a ZIP if the directory is larger than this, to avoid duplicating
#   the same bytes during zipping and triggering eviction.
MAX_JOB_BYTES = int(os.getenv("FMAP_MAX_JOB_BYTES", str(int(1.6 * 1024**3))))  # ~1.6GB
ZIP_MAX_BYTES = int(os.getenv("FMAP_ZIP_MAX_BYTES", str(int(900 * 1024**2))))  # ~900MB

os.makedirs(JOB_ROOT, exist_ok=True)

app = FastAPI(title="FMAP-AI API", version=APP_VERSION)

# Optional CORS (set FMAP_CORS_ORIGINS="https://yourdomain.com,https://other.com")
cors = os.getenv("FMAP_CORS_ORIGINS", "").strip()
if cors:
    origins = [o.strip() for o in cors.split(",") if o.strip()]
else:
    origins = []

allow_creds = os.getenv("FMAP_CORS_ALLOW_CREDENTIALS", "false").strip().lower() in ("1","true","yes")
if allow_creds and not origins:
    raise RuntimeError("FMAP_CORS_ORIGINS is required when FMAP_CORS_ALLOW_CREDENTIALS=true")


app.add_middleware(
    CORSMiddleware,
    # Allow calls from any frontend origin (including file:// and custom domains).
    # No cookies/credentials are used, so wildcard origins are safe here.
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory job store (sufficient for a single Render instance)
JOBS: Dict[str, Dict[str, Any]] = {}

# ----------------------------
# Debug / instrumentation
# ----------------------------
LOG_LEVEL = os.getenv("FMAP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s %(levelname)s %(message)s',
)
_logger = logging.getLogger("fmap")

# Optional debug key to protect debug/stack endpoints.
# If set, callers must send header: X-DEBUG-KEY: <value>
DEBUG_KEY = os.getenv("FMAP_DEBUG_KEY", "").strip()


def _require_debug_key(req: Request) -> None:
    if not DEBUG_KEY:
        return
    got = (req.headers.get("X-DEBUG-KEY") or "").strip()
    if got != DEBUG_KEY:
        raise HTTPException(status_code=401, detail="Missing/invalid debug key")


def _job_log_path(job_dir: str) -> str:
    return os.path.join(job_dir, "job.log")


def _append_job_log(job_dir: str, line: str) -> None:
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(_job_log_path(job_dir), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _job_event(job_id: str, stage: str, message: str = "", **extra: Any) -> None:
    meta = JOBS.get(job_id)
    job_dir = _job_dir(job_id)
    ts = _utc_now()

    if meta is not None:
        meta["stage"] = stage
        meta["message"] = message
        meta["last_update_at"] = ts
        meta["last_update_epoch"] = time.time()
        if extra:
            meta.setdefault("extra", {}).update(extra)

    line = f"{ts} [{job_id}] [{stage}] {message}".rstrip()
    _logger.info(line)
    _append_job_log(job_dir, line)


def _start_heartbeat(job_id: str, interval_seconds: int = 30) -> threading.Event:
    stop = threading.Event()

    def _beat():
        while not stop.wait(interval_seconds):
            meta = JOBS.get(job_id)
            if not meta:
                return
            if meta.get("status") in ("done", "error"):
                return
            meta["last_heartbeat_at"] = _utc_now()

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def _tail_text_file(path: str, max_lines: int = 200) -> str:
    try:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception:
        return ""



def _tmp_usage_mb() -> Dict[str, float]:
    """Best-effort /tmp usage snapshot (MB)."""
    try:
        total, used, free = shutil.disk_usage("/tmp")
        return {
            "tmp_total_mb": round(total / 1024 / 1024, 1),
            "tmp_used_mb": round(used / 1024 / 1024, 1),
            "tmp_free_mb": round(free / 1024 / 1024, 1),
        }
    except Exception:
        return {}


def _largest_files(job_dir: str, top_n: int = 8) -> List[Dict[str, Any]]:
    """Return largest files under job_dir (relative path + MB)."""
    rows: List[Dict[str, Any]] = []
    try:
        for root, _dirs, files in os.walk(job_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    sz = os.path.getsize(fp)
                except Exception:
                    continue
                rel = os.path.relpath(fp, job_dir)
                rows.append({"path": rel, "mb": round(sz / 1024 / 1024, 2)})
        rows.sort(key=lambda r: r["mb"], reverse=True)
        return rows[: max(1, min(int(top_n), 50))]
    except Exception:
        return []


def _clean_success_outputs(job_dir: str) -> None:
    """Remove heavy outputs after success (keeps job.log)."""
    keep = {"job.log"}
    try:
        for root, _dirs, files in os.walk(job_dir):
            for fn in files:
                if fn in keep:
                    continue
                fp = os.path.join(root, fn)
                try:
                    os.remove(fp)
                except Exception:
                    pass
    except Exception:
        pass



def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> str:
    return os.path.join(JOB_ROOT, job_id)


def _dir_size_bytes(path: str) -> int:
    """Return recursive directory size in bytes (best-effort)."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                try:
                    fp = os.path.join(root, fn)
                    total += os.path.getsize(fp)
                except Exception:
                    pass
    except Exception:
        return total
    return total



# -----------------------------------------------------------------------------
# Description file (human-readable catalog) per job
# -----------------------------------------------------------------------------
DESCRIPTION_FILENAME = os.getenv("FMAP_DESCRIPTION_FILENAME", "description.txt")


def _job_description_path(job_dir: str) -> str:
    return os.path.join(job_dir, DESCRIPTION_FILENAME)


def _classify_output_item(relpath: str) -> Dict[str, str]:
    """Heuristic classification for job output items."""
    p = relpath.lower()
    if p.endswith(".json") and "analysis_result" in p:
        return {"type": "analysis", "desc": "Main analysis summary consumed by the frontend (charts/cards)."}
    if p.endswith(".csv") and "climate_point" in p:
        return {"type": "download+clean", "desc": "GridMET point time series (downloaded via NCSS) after cleaning/units fixes."}
    if p.endswith(".csv") and "spi" in p:
        return {"type": "computed", "desc": "SPI time series computed from precipitation."}
    if p.endswith(".geojson"):
        return {"type": "download", "desc": "Vector output (e.g., fire perimeters) clipped to bbox."}
    if p.endswith(".tif") or p.endswith(".tiff"):
        if "ndvi" in p or "ndmi" in p or "nbr" in p:
            return {"type": "computed", "desc": "Vegetation index raster derived from Landsat over bbox."}
        if "nlcd" in p or "canopy" in p:
            return {"type": "download", "desc": "NLCD landcover/canopy raster fetched for bbox (used for forest mask/canopy)."}
        if "mtbs" in p or "burn" in p or "dnbr" in p or "severity" in p:
            return {"type": "download", "desc": "Burn severity / fire raster for bbox (when available)."}
        if "agb" in p or "carbon" in p or "fia" in p:
            return {"type": "download", "desc": "Biomass/carbon raster for bbox (point + forest distributions)."}
        return {"type": "download", "desc": "Raster output for bbox (source-dependent)."}
    if p.endswith(".log") or "state.json" in p:
        return {"type": "debug", "desc": "Debug metadata/logs for troubleshooting."}
    if p.endswith(".zip"):
        return {"type": "bundle", "desc": "ZIP bundle of job outputs."}
    if p.endswith(".txt") and "description" in p:
        return {"type": "documentation", "desc": "Human-readable description of outputs and chart meanings."}
    return {"type": "output", "desc": "Job output file."}


def _write_description_txt(job_id: str, job_dir: str, request_dict: Dict[str, Any]) -> str:
    """Create/update a per-job description.txt inside the job directory."""
    # Collect current files in job_dir (relative paths)
    rel_files: List[str] = []
    for root, _dirs, files in os.walk(job_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, job_dir)
            # Skip the job zip if present (can be huge and redundant)
            if rel.lower().endswith(".zip") and rel.startswith(job_id):
                continue
            rel_files.append(rel)
    rel_files = sorted(rel_files)

    # Build documentation
    lines: List[str] = []
    lines.append("FMAP-AI — Outputs & Chart Guide")
    lines.append(f"Job ID: {job_id}")
    lines.append(f"Generated (UTC): {_utc_now()}")
    lines.append("")
    lines.append("A) What this run does")
    lines.append("  - Downloads point/bbox subsets from public datasets (GridMET + optional remote-sensing layers).")
    lines.append("  - Computes derived indices (SPI, vegetation indices, fire-weather indices when enabled).")
    lines.append("  - Summarizes forest/disturbance context (landcover, canopy, biomass/carbon, fire products when available).")
    lines.append("  - Packages results for the web dashboard and optional ZIP download.")
    lines.append("")
    lines.append("B) Run inputs")
    for k in [
        "pt_lon","pt_lat","minlon","minlat","maxlon","maxlat",
        "date_start","date_end","spi_start","cloud_cover_lt",
        "keep_rasters","include_zip","size_px","selection"
    ]:
        if k in request_dict:
            lines.append(f"  - {k}: {request_dict[k]}")
    lines.append("")
    lines.append("C) Charts in the dashboard (how to interpret)")
    lines.append("  - Climate/Weather: precipitation + temperature series; optional SPI for drought context.")
    lines.append("  - Vegetation (Landsat): NDVI/NDMI/NBR indicate greenness, moisture stress, and burn/disturbance signal.")
    lines.append("  - Landcover/Canopy: forest mask and canopy % (if NLCD services are available).")
    lines.append("  - Burn/Fire: fire perimeters/burned area; burn severity if MTBS coverage exists.")
    lines.append("  - Biomass/Carbon: AGB/AGC point + forest-only distributions in the bbox.")
    lines.append("  - Fire/Forest Weather (GridMET-only, if enabled): KBDI, VPD, ETo, wind, HDW proxy, fuel moisture, ERC/BI.")
    lines.append("")
    lines.append("D) ZIP / output items produced by this run")
    if not rel_files:
        lines.append("  (No files found in job directory.)")
    else:
        for rel in rel_files:
            meta = _classify_output_item(rel)
            lines.append(f"  - {rel} | type={meta['type']} | {meta['desc']}")
    lines.append("")
    lines.append("E) Notes on missing/empty panels")
    lines.append("  - Some panels require specific data availability (e.g., NLCD WCS can be temporarily 503).")
    lines.append("  - Fire/burn panels can be empty if no fires intersect the bbox for the selected period.")
    lines.append("  - Distribution/histogram panels require bbox raster sampling; point-only runs may only show point/bbox mean.")
    lines.append("")

    path = _job_description_path(job_dir)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def _safe_rmtree(path: str) -> None:
    try:
        if path and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _cleanup_old_job_dirs() -> None:
    """Filesystem-based cleanup (works even after instance restarts and JOBS is empty)."""
    now = time.time()
    try:
        for name in os.listdir(JOB_ROOT):
            p = os.path.join(JOB_ROOT, name)
            if not os.path.isdir(p):
                continue
            try:
                age = now - os.path.getmtime(p)
            except Exception:
                age = now - now
            if age > MAX_JOB_AGE_SECONDS:
                _safe_rmtree(p)
    except Exception:
        pass


def _prune_job_dir(job_dir: str, keep_rasters: bool, keep_zip: bool) -> Dict[str, Any]:
    """Remove large intermediates to keep Render temp storage under control."""
    before = _dir_size_bytes(job_dir)

    # Always remove sampling intermediates
    _safe_rmtree(os.path.join(job_dir, "region_samples"))
    _safe_rmtree(os.path.join(job_dir, "__pycache__"))

    removed_files = 0
    removed_bytes = 0

    large_exts = (".tif", ".tiff", ".nc", ".hdf", ".h5", ".grib", ".grb", ".npy", ".npz")
    tmp_exts = (".tmp", ".part", ".cache")

    for root, _dirs, files in os.walk(job_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            lower = fn.lower()
            try:
                size = os.path.getsize(fp)
            except Exception:
                size = 0

            # Always remove obvious temp/cache artifacts
            if lower.endswith(tmp_exts):
                try:
                    os.remove(fp)
                    removed_files += 1
                    removed_bytes += size
                except Exception:
                    pass
                continue

            # Optionally remove large rasters/grids
            if (not keep_rasters) and lower.endswith(large_exts):
                try:
                    os.remove(fp)
                    removed_files += 1
                    removed_bytes += size
                except Exception:
                    pass
                continue

            # Optionally remove zip artifacts (rare; usually keep_zip is True only when include_zip)
            if (not keep_zip) and lower.endswith(".zip"):
                try:
                    os.remove(fp)
                    removed_files += 1
                    removed_bytes += size
                except Exception:
                    pass

    after = _dir_size_bytes(job_dir)
    return {
        "before_bytes": int(before),
        "after_bytes": int(after),
        "removed_files": int(removed_files),
        "removed_bytes": int(removed_bytes),
    }


def _cleanup_old_jobs() -> None:
    _cleanup_old_job_dirs()
    now = time.time()
    dead = []
    for jid, meta in list(JOBS.items()):
        started = meta.get("started_epoch", now)
        if (now - started) > MAX_JOB_AGE_SECONDS:
            dead.append(jid)

    for jid in dead:
        try:
            d = JOBS[jid].get("job_dir")
            if d and os.path.exists(d):
                _safe_rmtree(d)
        except Exception:
            pass
        JOBS.pop(jid, None)


class RunRequest(BaseModel):
    pt_lon: float = Field(..., description="Longitude (EPSG:4326)")
    pt_lat: float = Field(..., description="Latitude (EPSG:4326)")
    minlon: float
    minlat: float
    maxlon: float
    maxlat: float

    date_start: str = Field(..., description="YYYY-MM-DD")
    date_end: str = Field(..., description="YYYY-MM-DD")
    spi_start: str = Field("1981-01-01", description="YYYY-MM-DD")

    cloud_cover_lt: float = Field(30.0, ge=0.0, le=100.0, description="Landsat scene filter")

    # Output controls
    include_zip: bool = Field(False, description="If true, prepare a zip and expose /fmap/download/{job_id}")
    keep_rasters: bool = Field(False, description="If false, remove .tif/.nc after analysis (recommended on Render free tier)")

    # Selection mode:
    # - "point": compute time series for the single point (pt_lon/pt_lat)
    # - "region": approximate a region-mean time series by sampling multiple points inside the bbox/region and averaging
    selection: Literal["point", "region"] = Field("point", description="Select point or region mode (region uses sampling inside bbox/geojson).")

    # Optional region geometry (GeoJSON Feature or Geometry). If provided, sampling is restricted to the polygon.
    region_geojson: Optional[Dict[str, Any]] = Field(None, description="Optional GeoJSON Feature/Geometry to define a region. If omitted, bbox is used.")

    # Number of sample points used to approximate region-mean time series (only used when selection='region')
    region_n_samples: int = Field(9, ge=1, le=49, description="Number of sample points for region-mean time series (max 49).")

    # Sampling strategy inside bbox/polygon (only used when selection='region')
    region_sample_strategy: Literal["grid", "random"] = Field("grid", description="Sampling strategy for region time series: grid or random.")
    size_px: int = Field(900, ge=128, le=2000, description="Output raster size for ArcGIS/WCS exports")

    @model_validator(mode="after")
    def _validate_bbox(self):
        if self.minlon >= self.maxlon or self.minlat >= self.maxlat:
            raise ValueError("Invalid bbox: ensure min < max for lon/lat.")
        return self


class RunResponse(BaseModel):
    job_id: str
    status: str
    started_at: str
    result_url: str
    download_url: Optional[str] = None


class StatusResponse(BaseModel):
    job_id: str
    status: str
    stage: Optional[str] = None
    message: Optional[str] = None
    started_at: str
    finished_at: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    last_update_at: Optional[str] = None
    last_heartbeat_at: Optional[str] = None
    error: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    started_at: str
    finished_at: str
    analysis: Dict[str, Any]
    manifest: Dict[str, Any]
    download_url: Optional[str] = None
    description_url: Optional[str] = None


def _run_job(job_id: str, req: RunRequest) -> None:
    job_dir = _job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)

    JOBS[job_id].update(
        {
            "status": "running",
            "job_dir": job_dir,
            "started_at": _utc_now(),
            "started_epoch": time.time(),
        }
    )

    hb_stop = _start_heartbeat(job_id)
    _job_event(job_id, "job:start", "Job started")

    try:
        _job_event(job_id, "download:start", "Starting download pipeline")

        download_meta = run_download_pipeline(
            job_dir=job_dir,
            pt_lon=req.pt_lon,
            pt_lat=req.pt_lat,
            bbox_lonlat=(req.minlon, req.minlat, req.maxlon, req.maxlat),
            date_start=req.date_start,
            date_end=req.date_end,
            spi_start=req.spi_start,
            cloud_cover_lt=req.cloud_cover_lt,
            size_px=req.size_px,
        )

        # Storage guardrail (Render free instances can be evicted near ~2GB temp usage).
        bytes_after_download = _dir_size_bytes(job_dir)
        JOBS[job_id]["job_dir_bytes_after_download"] = int(bytes_after_download)

        # Record disk snapshot + biggest files (helps diagnose /tmp evictions).
        try:
            JOBS[job_id]["tmp_usage_after_download"] = _tmp_usage_mb()
            JOBS[job_id]["largest_files_after_download"] = _largest_files(job_dir, top_n=8)
        except Exception:
            pass

        if bytes_after_download > MAX_JOB_BYTES:
            # Try an emergency prune even if the caller requested keep_rasters=true.
            _job_event(
                job_id,
                "storage:limit",
                f"Outputs exceed guardrail after download ({bytes_after_download/1024**2:.1f} MB). Pruning large intermediates.",
            )
            prune_limit = _prune_job_dir(job_dir, keep_rasters=False, keep_zip=req.include_zip)
            JOBS[job_id]["prune_report_limit"] = prune_limit
            bytes_after_download = int(prune_limit.get("after_bytes", bytes_after_download))
            JOBS[job_id]["job_dir_bytes_after_download_pruned"] = int(bytes_after_download)
            try:
                JOBS[job_id]["tmp_usage_after_download_pruned"] = _tmp_usage_mb()
                JOBS[job_id]["largest_files_after_download_pruned"] = _largest_files(job_dir, top_n=8)
            except Exception:
                pass

            if bytes_after_download > MAX_JOB_BYTES:
                raise RuntimeError(
                    "Job outputs exceeded the instance storage guardrail after download "
                    f"({bytes_after_download/1024**2:.1f} MB). Reduce bbox/size_px, "
                    "set keep_rasters=false, or run on an instance with persistent disk."
                )

        _job_event(job_id, "download:done", "Download pipeline finished")

        _job_event(job_id, "analysis:start", "Starting analysis")
        analysis = run_analysis(job_dir=job_dir, request=req.model_dump(), download_meta=download_meta)
        _job_event(job_id, "analysis:done", "Analysis finished")

        # --- Optional time-series products (point and/or region) ---
        ts_warnings: List[str] = []
        try:
            if getattr(req, "selection", "point") == "region":
                # Build an approximate region-mean climate series by sampling multiple points inside the bbox/geojson.
                _compute_region_mean_timeseries(job_dir, req)
        except Exception as e:
            ts_warnings.append(f"Region time series generation failed: {e}")

        try:
            ts_info = _ensure_timeseries_jsons(job_dir, req)
            if ts_warnings:
                ts_info.setdefault("warnings", []).extend(ts_warnings)
            if isinstance(analysis, dict):
                analysis["timeseries"] = ts_info
        except Exception as e:
            if isinstance(analysis, dict):
                analysis["timeseries"] = {"available": False, "warnings": [f"Time series packaging failed: {e}"]}
        # Prune intermediates and (optionally) large rasters to keep temp storage under control.
        _job_event(job_id, "prune:start", "Pruning intermediates")
        prune_report = _prune_job_dir(job_dir, keep_rasters=req.keep_rasters, keep_zip=req.include_zip)
        _job_event(job_id, "prune:done", f"Prune complete (after_bytes={prune_report.get('after_bytes')})")
        JOBS[job_id]["prune_report"] = prune_report
        if prune_report.get("after_bytes", 0) > MAX_JOB_BYTES:
            raise RuntimeError(
                "Job outputs are still too large for this instance after pruning "
                f"({prune_report.get('after_bytes', 0)/1024**2:.1f} MB). "
                "Try keep_rasters=false, reduce bbox/size_px, or upgrade to a persistent-disk instance."
            )

# Write a human-readable description file for this job (used by the frontend as a link).
try:
    desc_path = _write_description_txt(job_id, job_dir, req.model_dump())
    JOBS[job_id]["description_path"] = desc_path
except Exception as e:
    _job_event(job_id, "description:warn", f"Failed to write description.txt: {e}")
        manifest = build_manifest(job_dir)

        # Optionally build zip for user download (zipping duplicates bytes temporarily).
        zip_path = None
        if req.include_zip:
            bytes_for_zip = _dir_size_bytes(job_dir)
            if bytes_for_zip > ZIP_MAX_BYTES:
                raise RuntimeError(
                    "Refusing to build ZIP because outputs are too large for safe zipping on this instance "
                    f"({bytes_for_zip/1024**2:.1f} MB). Try keep_rasters=false or reduce bbox/size_px."
                )
            _job_event(job_id, "zip:start", "Building outputs zip")
            zip_path = shutil.make_archive(os.path.join(job_dir, f"{job_id}_outputs"), "zip", job_dir)
            _job_event(job_id, "zip:done", "ZIP ready")

        _job_event(job_id, "job:done", "Job finished successfully")

        JOBS[job_id].update(
            {
                "status": "done",
                "finished_at": _utc_now(),
                "analysis": analysis,
                "manifest": manifest,
                "zip_path": zip_path,
            }
        )


        # Optional: reclaim disk immediately on success when no ZIP is requested.
        # Set FMAP_CLEAN_ON_SUCCESS=1 on Render to keep the service from being evicted due to /tmp growth.
        if (not req.include_zip) and os.getenv("FMAP_CLEAN_ON_SUCCESS", "0").strip().lower() in ("1", "true", "yes"):
            _job_event(job_id, "cleanup:success", "Cleaning success outputs (keeping job.log)")
            _clean_success_outputs(job_dir)
            try:
                JOBS[job_id]["tmp_usage_after_cleanup"] = _tmp_usage_mb()
                JOBS[job_id]["job_dir_bytes_after_cleanup"] = int(_dir_size_bytes(job_dir))
            except Exception:
                pass


    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc(limit=10)
        _job_event(job_id, "job:error", err)

        JOBS[job_id].update(
            {
                "status": "error",
                "finished_at": _utc_now(),
                "error": err,
                "traceback": tb,
            }
        )

        # Always try to reclaim disk if a job fails (prevents repeated evictions).
        try:
            _safe_rmtree(job_dir)
            JOBS[job_id]["job_dir_cleaned"] = True
        except Exception:
            pass


    finally:
        try:
            hb_stop.set()
            JOBS[job_id]["last_heartbeat_at"] = _utc_now()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}



@app.api_route("/health_head", methods=["GET", "HEAD"])
def health_head():
    return Response(status_code=200)

@app.get("/")
def root():
    return {"service": "FMAP-AI API", "version": APP_VERSION, "docs": "/docs"}


@app.head("/")
def root_head():
    return Response(status_code=200)


@app.post("/fmap/run", response_model=RunResponse)
def run(req: RunRequest, background: BackgroundTasks):
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "started_at": _utc_now(),
        "started_epoch": time.time(),
        "job_dir": _job_dir(job_id),
    }

    background.add_task(_run_job, job_id, req)

    download_url = f"/fmap/download/{job_id}" if req.include_zip else None
    return RunResponse(
        job_id=job_id,
        status=JOBS[job_id]["status"],
        started_at=JOBS[job_id]["started_at"],
        result_url=f"/fmap/result/{job_id}",
        download_url=download_url,
        description_url=description_url,
    )


@app.post("/fmap/run_sync", response_model=ResultResponse)
def run_sync(req: RunRequest):
    """Synchronous version (use only for small bboxes)."""
    _cleanup_old_jobs()

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "running",
        "started_at": _utc_now(),
        "started_epoch": time.time(),
        "job_dir": _job_dir(job_id),
    }
    _run_job(job_id, req)

    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=500, detail="Job missing")

    if meta["status"] == "error":
        raise HTTPException(status_code=500, detail={"error": meta.get("error"), "traceback": meta.get("traceback")})
    download_url = f"/fmap/download/{job_id}" if meta.get("zip_path") else None
    # Description file is always generated for successful jobs (if present).
    job_dir = meta.get("job_dir")
    desc_path = _job_description_path(job_dir) if job_dir else None
    description_url = f"/fmap/description/{job_id}" if (desc_path and os.path.exists(desc_path)) else None
    return ResultResponse(
        job_id=job_id,
        status=meta["status"],
        started_at=meta["started_at"],
        finished_at=meta["finished_at"],
        analysis=meta["analysis"],
        manifest=meta["manifest"],
        download_url=download_url,
    )


@app.get("/fmap/status/{job_id}", response_model=StatusResponse)
def status(job_id: str):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    elapsed = None
    try:
        elapsed = float(time.time() - float(meta.get("started_epoch", time.time())))
    except Exception:
        elapsed = None
    return StatusResponse(
        job_id=job_id,
        status=meta.get("status", "unknown"),
        stage=meta.get("stage"),
        message=meta.get("message"),
        started_at=meta.get("started_at"),
        finished_at=meta.get("finished_at"),
        elapsed_seconds=elapsed,
        last_update_at=meta.get("last_update_at"),
        last_heartbeat_at=meta.get("last_heartbeat_at"),
        error=meta.get("error"),
    )


@app.get("/fmap/result/{job_id}", response_model=ResultResponse)
def result(job_id: str):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    if meta.get("status") in ("queued", "running"):
        raise HTTPException(status_code=409, detail={"status": meta.get("status"), "message": "Not finished yet. Use /fmap/status."})

    if meta.get("status") == "error":
        raise HTTPException(status_code=500, detail={"error": meta.get("error"), "traceback": meta.get("traceback")})

    download_url = f"/fmap/download/{job_id}" if meta.get("zip_path") else None
    return ResultResponse(
        job_id=job_id,
        status=meta["status"],
        started_at=meta["started_at"],
        finished_at=meta["finished_at"],
        analysis=meta["analysis"],
        manifest=meta["manifest"],
        download_url=download_url,
    )


@app.get("/fmap/download/{job_id}")
def download(job_id: str):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    zip_path = meta.get("zip_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP not available for this job (run with include_zip=true).")

    return FileResponse(zip_path, media_type="application/zip", filename=os.path.basename(zip_path))




@app.get("/fmap/description/{job_id}")
def description(job_id: str):
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    if meta.get("status") in ("queued", "running"):
        raise HTTPException(status_code=409, detail={"status": meta.get("status"), "message": "Not finished yet. Use /fmap/status."})

    if meta.get("status") == "error":
        raise HTTPException(status_code=500, detail={"error": meta.get("error"), "traceback": meta.get("traceback")})

    job_dir = meta.get("job_dir") or _job_dir(job_id)
    path = _job_description_path(job_dir)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="description.txt not available for this job.")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=DESCRIPTION_FILENAME)

@app.get("/fmap/debug/{job_id}")
def debug_job(job_id: str, req: Request, tail: int = 200) -> Dict[str, Any]:
    """Lightweight debug endpoint: returns job meta + tail of per-job log."""
    _require_debug_key(req)
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    job_dir = meta.get("job_dir") or _job_dir(job_id)
    log_tail = _tail_text_file(_job_log_path(job_dir), max_lines=max(10, min(int(tail), 2000)))

    # Avoid returning huge blobs
    safe_meta = {k: v for k, v in meta.items() if k not in ("analysis", "manifest")}
    return {"job_id": job_id, "meta": safe_meta, "log_tail": log_tail}


@app.get("/fmap/stack/{job_id}")
def debug_stack(job_id: str, req: Request, max_chars: int = 60000) -> Dict[str, Any]:
    """Dump stack traces of all threads (useful when a job looks 'stuck').

    Security: if FMAP_DEBUG_KEY is set, callers must provide X-DEBUG-KEY.
    """
    _require_debug_key(req)
    meta = JOBS.get(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    buf = io.StringIO()
    try:
        faulthandler.dump_traceback(file=buf, all_threads=True)
        txt = buf.getvalue()
    except Exception as e:
        txt = f"Unable to dump stack traces: {e}"

    # Avoid returning huge payloads
    try:
        mc = int(max_chars)
    except Exception:
        mc = 60000
    txt = txt[-max(1000, min(mc, 200000)) :]

    safe_meta = {k: v for k, v in meta.items() if k not in ("analysis", "manifest")}
    return {"job_id": job_id, "meta": safe_meta, "stack": txt}

# ----------------------------
# Time series helpers / API
# ----------------------------

def _json_dumps(obj: Any) -> str:
    import json as _json
    return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def _read_csv_timeseries(path: str) -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


def _resample_timeseries(df: "pd.DataFrame", freq: str) -> "pd.DataFrame":
    out = df.copy()
    if "time" not in out.columns:
        return out
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out.set_index("time")

    if freq == "daily":
        return out.reset_index()

    # monthly aggregates (precip: sum, others: mean)
    if freq == "monthly":
        agg = {}
        for c in out.columns:
            if c.lower() in ("pr", "precip", "precipitation", "ppt"):
                agg[c] = "sum"
            else:
                agg[c] = "mean"
        outm = out.resample("MS").agg(agg)
        return outm.reset_index()

    raise ValueError(f"Unsupported freq: {freq}")


def _linreg_slope_per_year(t: "pd.Series", y: "pd.Series") -> Optional[float]:
    import numpy as np
    import pandas as pd
    mask = (~pd.isna(t)) & (~pd.isna(y))
    if mask.sum() < 5:
        return None
    tt = t[mask]
    yy = y[mask].astype(float)

    x = tt.map(lambda d: d.toordinal()).astype(float).to_numpy()
    yv = yy.to_numpy()
    try:
        b = np.polyfit(x, yv, 1)[0]
        return float(b * 365.25)  # per-year slope
    except Exception:
        return None


def _spi_event_stats(time_col: "pd.Series", spi: "pd.Series", thr: float) -> Dict[str, Any]:
    import pandas as pd
    mask = spi.astype(float) < thr
    mask = mask.fillna(False).to_numpy()

    n_days = int(mask.sum())
    events = 0
    max_dur = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur > 0:
                events += 1
                max_dur = max(max_dur, cur)
                cur = 0
    if cur > 0:
        events += 1
        max_dur = max(max_dur, cur)

    worst_val = None
    worst_date = None
    try:
        spi_float = spi.astype(float)
        idx = spi_float.idxmin()
        worst_val = None if pd.isna(spi_float.loc[idx]) else float(spi_float.loc[idx])
        worst_date = None if pd.isna(time_col.loc[idx]) else str(pd.to_datetime(time_col.loc[idx]).date())
    except Exception:
        pass

    return {
        "threshold": thr,
        "days_below": n_days,
        "event_count": events,
        "max_duration_days": int(max_dur),
        "worst_spi": worst_val,
        "worst_date": worst_date,
    }


def _build_timeseries_payload(climate_csv: str, spi_csv: Optional[str], kind: str, freq: str) -> Dict[str, Any]:
    import pandas as pd
    import numpy as np

    df = _read_csv_timeseries(climate_csv)

    if spi_csv and os.path.exists(spi_csv):
        spi = _read_csv_timeseries(spi_csv)
        spi_cols = [c for c in spi.columns if c.lower().startswith("spi")]
        if spi_cols:
            spi = spi[["time", spi_cols[0]]].rename(columns={spi_cols[0]: "spi30"})
            df = df.merge(spi, on="time", how="left")

    df = _resample_timeseries(df, freq=freq)
    if "time" not in df.columns:
        raise ValueError("No 'time' column found in climate time series.")

    summary: Dict[str, Any] = {
        "kind": kind,
        "freq": freq,
        "n": int(len(df)),
        "start": None,
        "end": None,
        "variables": {},
        "trends_per_year": {},
    }
    if len(df) > 0:
        summary["start"] = str(pd.to_datetime(df["time"].iloc[0]).date())
        summary["end"] = str(pd.to_datetime(df["time"].iloc[-1]).date())

    for c in [c for c in df.columns if c != "time"]:
        s = pd.to_numeric(df[c], errors="coerce")
        summary["variables"][c] = {
            "mean": None if s.dropna().empty else float(s.mean()),
            "std": None if s.dropna().empty else float(s.std(ddof=1)),
            "min": None if s.dropna().empty else float(s.min()),
            "p10": None if s.dropna().empty else float(s.quantile(0.10)),
            "median": None if s.dropna().empty else float(s.quantile(0.50)),
            "p90": None if s.dropna().empty else float(s.quantile(0.90)),
            "max": None if s.dropna().empty else float(s.max()),
            "missing": int(s.isna().sum()),
        }
        slope = _linreg_slope_per_year(df["time"], s)
        if slope is not None:
            summary["trends_per_year"][c] = slope

    events = {}
    if freq == "daily" and "spi30" in df.columns:
        spi = pd.to_numeric(df["spi30"], errors="coerce")
        for thr in (-1.0, -1.5, -2.0):
            events[str(thr)] = _spi_event_stats(df["time"], spi, thr)
    summary["drought_events"] = events

    out_vars = {}
    for c in [c for c in df.columns if c != "time"]:
        arr = df[c].replace({np.nan: None}).tolist()
        out_vars[c] = [None if v is None else float(v) for v in arr]

    payload = {
        "meta": summary,
        "time": [str(pd.to_datetime(t).date()) for t in df["time"].tolist()],
        "variables": out_vars,
    }
    return payload


def _ensure_timeseries_jsons(job_dir: str, req: RunRequest) -> Dict[str, Any]:
    job_id = os.path.basename(job_dir.rstrip("/"))
    info: Dict[str, Any] = {"available": False, "point": {}, "region": {}}

    candidates = {
        "point": {
            "climate": os.path.join(job_dir, "climate_point_clean.csv"),
            "spi": os.path.join(job_dir, "spi30_point.csv"),
        },
        "region": {
            "climate": os.path.join(job_dir, "climate_region_mean.csv"),
            "spi": os.path.join(job_dir, "spi30_region.csv"),
        },
    }

    for kind in ("point", "region"):
        clim = candidates[kind]["climate"]
        spi = candidates[kind]["spi"]
        if not os.path.exists(clim):
            continue

        info[kind] = {"files": {}, "api": {}}
        for freq in ("daily", "monthly"):
            out_json = os.path.join(job_dir, f"timeseries_{kind}_{freq}.json")
            try:
                payload = _build_timeseries_payload(climate_csv=clim, spi_csv=spi, kind=kind, freq=freq)
                with open(out_json, "w", encoding="utf-8") as f:
                    f.write(_json_dumps(payload))
                info[kind]["files"][freq] = os.path.basename(out_json)
                info[kind]["api"][freq] = f"/fmap/timeseries/{job_id}?kind={kind}&freq={freq}"
            except Exception as e:
                info.setdefault("warnings", []).append(f"Failed to build {kind} {freq} time series: {e}")

    info["available"] = bool(info.get("point") or info.get("region"))
    return info


def _compute_region_mean_timeseries(job_dir: str, req: RunRequest) -> None:
    """
    Approximate region-mean daily climate series by sampling multiple points inside the bbox/region and averaging.
    Writes:
      - climate_region_mean.csv
      - spi30_region.csv
    """
    import math
    import random
    import pandas as pd

    bbox = (req.minlon, req.minlat, req.maxlon, req.maxlat)
    start_date = req.date_start
    end_date = req.date_end

    n_samples = int(getattr(req, "region_n_samples", 9) or 9)
    strategy = getattr(req, "region_sample_strategy", "grid") or "grid"
    region_geojson = getattr(req, "region_geojson", None)

    points: List[Tuple[float, float]] = []

    poly = None
    if region_geojson:
        try:
            from shapely.geometry import shape
            if isinstance(region_geojson, dict) and region_geojson.get("type") == "Feature":
                geom = region_geojson.get("geometry")
            else:
                geom = region_geojson
            poly = shape(geom) if geom else None
        except Exception:
            poly = None

    if strategy == "grid":
        ngrid = int(math.ceil(math.sqrt(n_samples)))
        if ngrid < 2:
            ngrid = 2
        lons = [bbox[0] + (bbox[2] - bbox[0]) * i / (ngrid - 1) for i in range(ngrid)]
        lats = [bbox[1] + (bbox[3] - bbox[1]) * j / (ngrid - 1) for j in range(ngrid)]
        for lat in lats:
            for lon in lons:
                if poly is not None:
                    try:
                        from shapely.geometry import Point as ShpPoint
                        if not poly.contains(ShpPoint(lon, lat)):
                            continue
                    except Exception:
                        pass
                points.append((lon, lat))
        if len(points) < 1:
            # fallback: ignore polygon filter
            points = [(lon, lat) for lat in lats for lon in lons]
        points = points[:n_samples]
    else:
        tries = 0
        while len(points) < n_samples and tries < n_samples * 300:
            tries += 1
            lon = random.uniform(bbox[0], bbox[2])
            lat = random.uniform(bbox[1], bbox[3])
            if poly is not None:
                try:
                    from shapely.geometry import Point as ShpPoint
                    if not poly.contains(ShpPoint(lon, lat)):
                        continue
                except Exception:
                    pass
            points.append((lon, lat))

    if not points:
        raise ValueError("No sample points could be generated for region time series.")

    sample_dir = os.path.join(job_dir, "region_samples")
    os.makedirs(sample_dir, exist_ok=True)

    merged: Optional[pd.DataFrame] = None

    for dataset_id, var_candidates in GRIDMET_DATASETS.items():
        per_var: Optional[pd.DataFrame] = None
        ok_points = 0

        for i, (lon, lat) in enumerate(points):
            out_csv = os.path.join(sample_dir, f"{dataset_id}_p{i+1}.csv")
            try:
                _, df_pt, _used_var = ncss_point_csv(
                    dataset_id=dataset_id,
                    var_candidates=var_candidates,
                    lon=lon,
                    lat=lat,
                    start_date=start_date,
                    end_date=end_date,
                    out_csv=out_csv,
                )
                vc = [c for c in df_pt.columns if c != "time"]
                if not vc:
                    continue
                df_pt = df_pt.rename(columns={vc[0]: f"{dataset_id}_p{i+1}"})
                df_pt["time"] = pd.to_datetime(df_pt["time"], errors="coerce", utc=True).dt.tz_convert(None)
                df_pt = df_pt.dropna(subset=["time"])
                per_var = df_pt if per_var is None else per_var.merge(df_pt, on="time", how="outer")
                ok_points += 1
            except Exception:
                continue

        if per_var is None or ok_points == 0:
            continue

        per_var = per_var.sort_values("time")
        cols = [c for c in per_var.columns if c.startswith(f"{dataset_id}_p")]
        per_var[dataset_id] = per_var[cols].astype(float).mean(axis=1, skipna=True)
        keep = per_var[["time", dataset_id]]
        merged = keep if merged is None else merged.merge(keep, on="time", how="outer")

    if merged is None or merged.empty:
        raise ValueError("Region climate time series could not be built (no datasets downloaded).")

    merged = merged.sort_values("time")
    out_clim = os.path.join(job_dir, "climate_region_mean.csv")
    merged.to_csv(out_clim, index=False)

    if "pr" in merged.columns:
        pr = pd.to_numeric(merged["pr"], errors="coerce")
        pr_series = pd.Series(pr.values, index=pd.to_datetime(merged["time"]))
        spi = spi_gamma_monthly(pr_series, scale_days=30)
        spi_df = pd.DataFrame({"time": spi.index, "spi30": spi.values})
        spi_df["time"] = pd.to_datetime(spi_df["time"], utc=True).dt.tz_convert(None)
        out_spi = os.path.join(job_dir, "spi30_region.csv")
        spi_df.to_csv(out_spi, index=False)


@app.get("/fmap/timeseries/{job_id}")
def fmap_timeseries(job_id: str, kind: str = "point", freq: str = "daily") -> Response:
    kind = (kind or "point").lower()
    freq = (freq or "daily").lower()
    if kind not in ("point", "region"):
        raise HTTPException(status_code=400, detail="kind must be 'point' or 'region'")
    if freq not in ("daily", "monthly"):
        raise HTTPException(status_code=400, detail="freq must be 'daily' or 'monthly'")

    job_dir = os.path.join(JOB_ROOT, job_id)
    path = os.path.join(job_dir, f"timeseries_{kind}_{freq}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Time series not found for this job/kind/freq.")

    with open(path, "r", encoding="utf-8") as f:
        return Response(content=f.read(), media_type="application/json")
