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
from typing import Any, Dict, Optional
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from fmap_download import run_download_pipeline
from fmap_analysis import run_analysis, build_manifest

APP_VERSION = "0.1.1"
JOB_ROOT = os.getenv("FMAP_JOB_ROOT", "/tmp/fmap_jobs")
MAX_JOB_AGE_SECONDS = int(os.getenv("FMAP_MAX_JOB_AGE_SECONDS", "86400"))  # 24h

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
    allow_origins=[
        "https://fmap.agrimetsoft.com",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,     # set False if you don’t use cookies/auth
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple in-memory job store (sufficient for a single Render instance)
JOBS: Dict[str, Dict[str, Any]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> str:
    return os.path.join(JOB_ROOT, job_id)


def _cleanup_old_jobs() -> None:
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
                shutil.rmtree(d, ignore_errors=True)
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
    keep_rasters: bool = Field(True, description="If false, remove .tif/.nc after analysis")
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
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None


class ResultResponse(BaseModel):
    job_id: str
    status: str
    started_at: str
    finished_at: str
    analysis: Dict[str, Any]
    manifest: Dict[str, Any]
    download_url: Optional[str] = None


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

    try:
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

        analysis = run_analysis(job_dir=job_dir, request=req.model_dump(), download_meta=download_meta)
        manifest = build_manifest(job_dir)

        # Optionally remove large rasters
        if not req.keep_rasters:
            for fn in os.listdir(job_dir):
                if fn.lower().endswith((".tif", ".tiff", ".nc")):
                    try:
                        os.remove(os.path.join(job_dir, fn))
                    except Exception:
                        pass
            manifest = build_manifest(job_dir)

        # Optionally build zip for user download
        zip_path = None
        if req.include_zip:
            zip_path = shutil.make_archive(os.path.join(job_dir, f"{job_id}_outputs"), "zip", job_dir)

        JOBS[job_id].update(
            {
                "status": "done",
                "finished_at": _utc_now(),
                "analysis": analysis,
                "manifest": manifest,
                "zip_path": zip_path,
            }
        )

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc(limit=10)
        JOBS[job_id].update(
            {
                "status": "error",
                "finished_at": _utc_now(),
                "error": err,
                "traceback": tb,
            }
        )


@app.get("/health")
def health():
    return {"ok": True, "version": APP_VERSION}


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
    return StatusResponse(
        job_id=job_id,
        status=meta.get("status", "unknown"),
        started_at=meta.get("started_at"),
        finished_at=meta.get("finished_at"),
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
