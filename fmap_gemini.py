"""
fmap_gemini.py
Server-side helper for calling Google Gemini via REST.

IMPORTANT:
- Do NOT put API keys in index.html (browser). Keep them as env vars on Render.
- This module is safe to import (no side effects / prints).
"""

from __future__ import annotations

import os
import requests
from typing import Any, Dict, Optional


class GeminiError(RuntimeError):
    pass


def _default_model() -> str:
    # You can override with Render env var GEMINI_MODEL
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def gemini_generate_text(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    timeout: int = 60,
) -> str:
    """
    Call Gemini REST API and return plain text.
    Uses Generative Language API (v1beta): models/{model}:generateContent
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError("Missing GEMINI_API_KEY (set it in Render Environment Variables).")

    model = model or _default_model()

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise GeminiError(f"Gemini HTTP {r.status_code}: {r.text[:500]}")

    data = r.json()

    # Extract text safely
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            raise GeminiError(f"Gemini returned no candidates: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        out = "\n".join([t for t in texts if t]).strip()
        if not out:
            raise GeminiError(f"Gemini returned empty text: {data}")
        return out
    except Exception as e:
        if isinstance(e, GeminiError):
            raise
        raise GeminiError(f"Failed to parse Gemini response: {e}")


# Backward-compatible aliases (in case main.py imports one of these names)
def call_gemini(prompt: str, api_key: Optional[str] = None, model: Optional[str] = None) -> str:
    return gemini_generate_text(prompt, api_key=api_key, model=model)

def generate_summary(prompt: str, api_key: Optional[str] = None, model: Optional[str] = None) -> str:
    return gemini_generate_text(prompt, api_key=api_key, model=model)
