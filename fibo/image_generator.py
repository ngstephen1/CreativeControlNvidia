# fibo/image_generator.py

import os
from pathlib import Path
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont


class FIBOImageGeneratorStub:
    """
    TEMPORARY STUB:
    Generates simple placeholder images from FIBO JSON payloads.

    Later, you can replace the core of this class with real FIBO model calls
    (local MLX or hosted API). The rest of your app won't need to change.
    """

    def __init__(self, output_dir: str = "generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _shorten(self, text: str, max_len: int = 80) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def generate_image_from_fibo_json(self, fibo_payload: Dict[str, Any]) -> str:
        """
        Create a simple placeholder PNG file that encodes:
          - scene_id, shot_id
          - short description
          - key camera params

        Returns: file path as string.
        """
        meta = fibo_payload.get("meta", {})
        scene_id = meta.get("scene_id", 1)
        shot_id = meta.get("shot_id", "UNKNOWN")
        description = self._shorten(meta.get("original_description", ""))

        camera = fibo_payload.get("camera", {})
        angle = camera.get("angle", "eye-level")
        fov = camera.get("fov", 50)
        focal_length = camera.get("focal_length", "50mm")

        # simple 1024x576 placeholder
        width, height = 1024, 576
        img = Image.new("RGB", (width, height), color=(20, 20, 30))
        draw = ImageDraw.Draw(img)

        # Safe default font
        try:
            font_title = ImageFont.truetype("Arial.ttf", 36)
            font_body = ImageFont.truetype("Arial.ttf", 24)
        except Exception:
            font_title = ImageFont.load_default()
            font_body = ImageFont.load_default()

        title = f"Scene {scene_id} - Shot {shot_id}"
        camera_line = f"Camera: {angle}, FOV {fov}, {focal_length}"

        # Positions
        draw.text((40, 40), title, fill=(255, 255, 255), font=font_title)
        draw.text((40, 100), description, fill=(200, 200, 200), font=font_body)
        draw.text((40, 150), camera_line, fill=(180, 180, 220), font=font_body)

        # maybe a simple framing rectangle to hint "storyboard frame"
        margin = 20
        draw.rectangle(
            [margin, margin, width - margin, height - margin],
            outline=(90, 90, 130),
            width=4,
        )

        filename = f"scene{scene_id}_shot{shot_id}.png"
        filepath = str(self.output_dir / filename)
        img.save(filepath)

        return filepath