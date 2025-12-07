import os
import uuid
import json
from typing import Any, Dict

import requests


class FIBOBriaImageGenerator:
    """
    Wrapper around Bria v2 /image/generate endpoint, using our FIBO-style JSON
    as the `structured_prompt`.

    Docs:
      - Base URL: https://engine.prod.bria-api.com
      - Endpoint: POST /v2/image/generate
      - Auth header: api_token: <your API key>
      - Response (sync): includes result.image_url, result.seed, result.structured_prompt
    """

    def __init__(self, api_key: str, output_dir: str = "generated") -> None:
        if not api_key:
            raise ValueError("Bria API key must not be empty")

        self.api_key = api_key
        self.base_url = "https://engine.prod.bria-api.com/v2/image/generate"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_image_from_fibo_json(self, fibo_json: Dict[str, Any]) -> str:
        """
        Send a structured_prompt (FIBO JSON) to Bria and download the resulting image.

        Returns:
            Local file path to the downloaded PNG.
        """
        payload: Dict[str, Any] = {
            # Bria expects structured_prompt to be a STRING,
            # usually a JSON string describing the scene.
            "structured_prompt": json.dumps(fibo_json),
            "sync": True,
        }

        headers = {
            "Content-Type": "application/json",
            "api_token": self.api_key,
        }

        # 1) Call Bria image generation API
        resp = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=60,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Bria API error {resp.status_code}: {resp.text}"
            )

        data = resp.json()

        # Sync v2 responses usually look like:
        # { "status": "COMPLETED", "result": { "image_url": "...", "seed": ..., "structured_prompt": ... } }
        result = data.get("result", data)
        image_url = result.get("image_url")

        if not image_url:
            raise RuntimeError(
                f"No image_url found in Bria response: {data}"
            )

        # 2) Download the image to local disk
        img_resp = requests.get(image_url, timeout=60)
        if img_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to download image from {image_url}: "
                f"{img_resp.status_code} {img_resp.text}"
            )

        filename = f"fibo_{uuid.uuid4().hex}.png"
        out_path = os.path.join(self.output_dir, filename)

        with open(out_path, "wb") as f:
            f.write(img_resp.content)

        return out_path