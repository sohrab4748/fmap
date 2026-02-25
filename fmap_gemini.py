"""
fmap_gemini.py

Gemini helper functions for FMAP.
- Safe to import (no prints / no side effects).
- Uses GEMINI_API_KEY from environment (Render Environment Variables).
- Provides interpret_with_gemini() expected by main.py.
"""

from __future__ import annotations

import os
import requests
from typing import Any, Dict, Optional


class GeminiError(RuntimeError):
    """Raised when Gemini call fails."""


def _get_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise GeminiError("Missing GEMINI_API_KEY. Set it in Render → Environment Variables.")
    return key


def _get_model(model: Optional[str] = None) -> str:
    # You can override in Render with GEMINI_MODEL
    return model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def gemini_generate_text(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
    timeout: int = 90,
) -> str:
    """
    Call Gemini REST API (Generative Language API v1beta) and return plain text.
    """
    key = _get_api_key(api_key)
    mdl = _get_model(model)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent?key={key}"
    payload: Dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }

    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise GeminiError(f"Gemini HTTP {r.status_code}: {r.text[:800]}")

    data = r.json()

    # Extract text from first candidate
    candidates = data.get("candidates", [])
    if not candidates:
        raise GeminiError(f"Gemini returned no candidates: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    out = "\n".join([t for t in texts if t]).strip()
    if not out:
        raise GeminiError(f"Gemini returned empty text: {data}")

    return out


def interpret_with_gemini(
    prompt: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 1024,
) -> str:
    """
    Backward-compatible entrypoint expected by your main.py.
    """
    return gemini_generate_text(
        prompt,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
