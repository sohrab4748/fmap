"""Gemini interpretation helper.

IMPORTANT: Do not call Gemini directly from the browser. Keep GEMINI_API_KEY on the server.
Uses Google Gen AI SDK (google-genai).
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from google import genai


def _compact_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce analysis JSON to a compact payload for LLM prompting."""
    keep = {}
    for k in ["point", "bbox", "landcover", "vegetation", "climate", "spi30", "disturbance", "biomass_carbon", "carbon_loss_proxy"]:
        if k in analysis:
            keep[k] = analysis[k]
    # Avoid huge manifests/download_meta if present
    return keep


def interpret_with_gemini(
    analysis: Dict[str, Any],
    mode: str = "technical",
    model: str = "gemini-3-flash-preview",
    extra: Optional[str] = None,
) -> str:
    """Return a narrative interpretation of FMAP analysis using Gemini."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set on the server (Render environment variable).")

    client = genai.Client()  # reads GEMINI_API_KEY from environment

    mode = (mode or "technical").strip().lower()
    if mode not in {"technical", "executive", "public"}:
        mode = "technical"

    compact = _compact_analysis(analysis)
    payload = json.dumps(compact, ensure_ascii=False, indent=2)

    style = {
        "technical": (
            "Write a technical forestry/remote-sensing interpretation for a scientist/analyst. "
            "Use units, mention data sources (Landsat, NLCD, gridMET, FIA/USFS, WFIGS/MTBS), and include caveats."
        ),
        "executive": (
            "Write an executive summary for a decision maker. "
            "Keep it concise with bullets and clear takeaways and risks."
        ),
        "public": (
            "Write a plain-language explanation for a general audience. "
            "Avoid jargon, explain terms briefly, and highlight uncertainties."
        ),
    }[mode]

    system = (
        "You are FMAP-AI, a forestry analysis assistant. "
        "Be accurate and conservative. If the point is not forest (NLCD non-forest), "
        "clearly state that forest-only metrics refer to nearby forest pixels within the bbox. "
        "Do not claim official wildfire impacts unless clearly supported by disturbance fields. "
        "If burned area is zero or unknown, say so."
    )

    prompt = f"""{system}

Task:
{style}

Output format:
- Title (one line)
- Summary (3–6 bullets)
- Forest / Vegetation condition (NDVI/NDMI/NBR; canopy; landcover)
- Biomass & carbon (AGB/AGC; include forest-only stats if present)
- Climate context for the selected window (precip, temperature, VPD, SPI-30)
- Disturbance signals (WFIGS perimeters, MTBS severity) with uncertainty notes
- Data quality / limitations (cloud cover, bbox size, non-forest point caveat, resolution mismatches)
- Recommended next steps (2–5 items)

Analysis JSON:
{payload}
"""

    if extra:
        prompt += "\n\nAdditional user instructions:\n" + str(extra).strip()

    resp = client.models.generate_content(model=model, contents=prompt)
    # The SDK exposes resp.text as convenience
    return getattr(resp, "text", None) or str(resp)
