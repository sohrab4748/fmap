# FMAP-AI Backend API (Render-ready)

This repo is a FastAPI backend that:

1) runs the FMAP download pipeline (your V12 logic, refactored into `fmap_download.py`)
2) runs analysis on the downloaded outputs (`fmap_analysis.py`)
3) returns JSON to your .NET server, and optionally provides a ZIP of outputs

## Endpoints

- `GET /health` → basic health check
- `POST /fmap/run` → async job (returns `job_id`)
- `GET /fmap/status/{job_id}` → job state
- `GET /fmap/result/{job_id}` → final JSON (analysis + manifest)
- `GET /fmap/download/{job_id}` → ZIP (only if you ran with `include_zip=true`)
- `POST /fmap/run_sync` → synchronous (use only for small bboxes)

## Request body example

```json
{
  "pt_lon": -93.62,
  "pt_lat": 42.03,
  "minlon": -93.70,
  "minlat": 41.99,
  "maxlon": -93.54,
  "maxlat": 42.09,
  "date_start": "2022-06-01",
  "date_end": "2022-06-30",
  "spi_start": "1981-01-01",
  "cloud_cover_lt": 30,
  "include_zip": true,
  "keep_rasters": true,
  "size_px": 900
}
```

## .NET (HttpClient) example

```csharp
using System.Net.Http.Json;

var http = new HttpClient();
http.BaseAddress = new Uri("https://YOUR-RENDER-SERVICE.onrender.com");

var req = new {
  pt_lon = -93.62, pt_lat = 42.03,
  minlon = -93.70, minlat = 41.99, maxlon = -93.54, maxlat = 42.09,
  date_start = "2022-06-01", date_end = "2022-06-30",
  spi_start = "1981-01-01",
  include_zip = true
};

// async job
var run = await http.PostAsJsonAsync("/fmap/run", req);
var runObj = await run.Content.ReadFromJsonAsync<Dictionary<string, object>>();
var jobId = runObj["job_id"].ToString();

// poll
while(true)
{
  var st = await http.GetFromJsonAsync<Dictionary<string, object>>($"/fmap/status/{jobId}");
  if(st["status"].ToString() == "done") break;
  if(st["status"].ToString() == "error") throw new Exception(st["error"].ToString());
  await Task.Delay(2000);
}

// result
var result = await http.GetStringAsync($"/fmap/result/{jobId}");
// optional zip: GET /fmap/download/{jobId}
```

## Render deployment

This repo includes `render.yaml` (Blueprint). Connect your GitHub repo to Render and deploy.

Environment variables:
- `FMAP_CORS_ORIGINS`: comma-separated list (or `*`)
- `FMAP_MAX_JOB_AGE_SECONDS`: cleanup threshold
- `FMAP_JOB_ROOT`: where job outputs are written (default `/tmp/fmap_jobs`)


## CORS

If you call this API from a browser (e.g., `https://fmap.agrimetsoft.com`), set:

- `FMAP_CORS_ORIGINS=https://fmap.agrimetsoft.com`

Optional (only if you truly need cookies/credentials):

- `FMAP_CORS_ALLOW_CREDENTIALS=true` (requires `FMAP_CORS_ORIGINS` to be explicitly set)

## gridMET decoding

The gridMET THREDDS/NCSS CSV responses can return **packed integer values**. This backend decodes them using each variable's `scale_factor` and `add_offset` (cached), so `climate_point_clean.csv` contains realistic units:

- precipitation → mm
- tmmx/tmmn → °C
- vpd → kPa
