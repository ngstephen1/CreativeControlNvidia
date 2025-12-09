# app/bria_tools.py

"""
Thin wrappers around Bria HTTP APIs (Upscale + RMBG).

These do NOT run any local GPU models – they just:
  * read an input image from disk
  * POST to Bria's cloud endpoint
  * return the resulting image bytes

Configure the actual API URLs + auth in your .env:
  BRIA_API_TOKEN=...
  BRIA_UPSCALE_ENDPOINT=https://...   # from Bria docs
  BRIA_RMBG_ENDPOINT=https://...      # from Bria docs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

BRIA_API_TOKEN = os.getenv("BRIA_API_TOKEN")
BRIA_UPSCALE_ENDPOINT = os.getenv("BRIA_UPSCALE_ENDPOINT")
BRIA_RMBG_ENDPOINT = os.getenv("BRIA_RMBG_ENDPOINT")


class BriaConfigError(RuntimeError):
    """Raised when required Bria config/env vars are missing."""


def _auth_headers() -> Dict[str, str]:
    """
    Build auth headers for Bria.

    Adjust the header key/value to match Bria's REST docs
    (e.g. 'Authorization: Bearer <token>' or 'x-api-key: <token>').
    """
    if not BRIA_API_TOKEN:
        raise BriaConfigError("BRIA_API_TOKEN is not set in the environment (.env).")

    return {
        "Authorization": f"Bearer {BRIA_API_TOKEN}",  # change if Bria uses a different scheme
    }


def _post_image(
    endpoint: str,
    image_bytes: bytes,
    extra_fields: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> bytes:
    """Generic helper to POST an image to a Bria endpoint and return the result bytes."""
    if not endpoint:
        raise BriaConfigError("Bria endpoint URL is not configured.")

    files = {
        # Bria's API may expect a specific field name; adjust "image" if needed.
        "image": ("input.png", image_bytes, "image/png"),
    }
    data = extra_fields or {}

    resp = requests.post(
        endpoint,
        headers=_auth_headers(),
        files=files,
        data=data,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        # Include a small prefix of the response text for debugging
        snippet = resp.text[:500]
        raise RuntimeError(f"Bria API error {resp.status_code}: {snippet}")

    return resp.content


def upscale_image_file(path: Path, scale: int = 2) -> bytes:
    """
    Call Bria Upscale API for a local file.

    Args:
        path: Path to the input image file.
        scale: Upscale factor (2, 4, etc.) – pass through to the API payload.

    Returns:
        Output image bytes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    if not BRIA_UPSCALE_ENDPOINT:
        raise BriaConfigError("BRIA_UPSCALE_ENDPOINT is not configured in .env.")

    with path.open("rb") as f:
        img_bytes = f.read()

    extra = {"scale": str(scale)}  # adjust keys according to Bria's docs
    return _post_image(BRIA_UPSCALE_ENDPOINT, img_bytes, extra_fields=extra)


def rmbg_image_file(path: Path) -> bytes:
    """
    Call Bria RMBG (background removal) API for a local file.

    Args:
        path: Path to input image.

    Returns:
        Output image bytes (likely PNG with alpha).
    """
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    if not BRIA_RMBG_ENDPOINT:
        raise BriaConfigError("BRIA_RMBG_ENDPOINT is not configured in .env.")

    with path.open("rb") as f:
        img_bytes = f.read()

    # No extra fields by default; add if Bria's docs specify options.
    return _post_image(BRIA_RMBG_ENDPOINT, img_bytes)