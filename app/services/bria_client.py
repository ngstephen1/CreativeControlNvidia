# app/services/bria_client.py

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


class BriaClientError(Exception):
    """Simple wrapper for Bria API errors."""


class BriaClient:
    BASE_URL = "https://engine.prod.bria-api.com/v2"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("BRIA_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "BRIA_API_KEY is not set. Add it to your .env or environment variables."
            )

    async def generate_image_from_prompt(
        self,
        prompt: str,
        *,
        seed: Optional[int] = None,
        sync: bool = True,
        ip_signal: bool = False,
        prompt_content_moderation: bool = True,
        visual_input_content_moderation: bool = True,
    ) -> Dict[str, Any]:
        """
        Call Bria v2 /image/generate API with a text prompt.

        Returns a dict containing:
          - image_url (str)
          - structured_prompt (dict or None)
          - seed (int)
          - raw_response (full JSON)
        """
        url = f"{self.BASE_URL}/image/generate"

        headers = {
            "Content-Type": "application/json",
            "api_token": self.api_key,
        }

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "sync": sync,
            "ip_signal": ip_signal,
            "prompt_content_moderation": prompt_content_moderation,
            "visual_input_content_moderation": visual_input_content_moderation,
        }
        if seed is not None:
            payload["seed"] = seed

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            try:
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise BriaClientError(f"Non-JSON response from Bria: {resp.text}") from exc

        if resp.status_code >= 400:
            raise BriaClientError(
                f"Bria error {resp.status_code}: {data}"
            )

        # Bria v2 with sync=true typically returns { status, result: { image_url, seed, ... } }
        result = data.get("result", data)

        image_url = result.get("image_url")
        seed_out = result.get("seed")
        structured_prompt = result.get("structured_prompt")

        return {
            "image_url": image_url,
            "seed": seed_out,
            "structured_prompt": structured_prompt,
            "raw_response": data,
        }