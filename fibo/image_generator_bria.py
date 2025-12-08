import os
import uuid
import json
from typing import Any, Dict, Optional

import requests


# ---------------------------------------------------------------------------
# Helper preset dictionaries
# ---------------------------------------------------------------------------

CAMERA_PRESETS: Dict[str, Dict[str, Any]] = {
    # Designed for music-video / cinematic language
    "mv_anamorphic_35mm": {
        "lens_focal_length": "35mm anamorphic",
        "depth_of_field": "shallow",
        "shutter_speed": "1/48",
        "motion_blur": "cinematic subtle",
        "notes": "Classic anamorphic look, oval bokeh, strong sense of depth.",
    },
    "mv_telephoto_closeup": {
        "lens_focal_length": "85mm",
        "depth_of_field": "very shallow",
        "shutter_speed": "1/100",
        "motion_blur": "minimal",
        "notes": "Flattering portrait closeups, strong subject isolation.",
    },
    "mv_wide_city_scope": {
        "lens_focal_length": "24mm",
        "depth_of_field": "medium",
        "shutter_speed": "1/60",
        "motion_blur": "slight",
        "notes": "Wide establishing frames with city context and leading lines.",
    },
    "mv_handheld_docu": {
        "lens_focal_length": "28mm",
        "depth_of_field": "medium",
        "shutter_speed": "1/80",
        "motion_blur": "handheld",
        "notes": "Handheld feel, documentary-style realism.",
    },
}

LIGHTING_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "neon_rain_mv": {
        "conditions": (
            "nighttime, wet streets, practical neon signs and street lamps "
            "creating reflections in puddles"
        ),
        "direction": (
            "mixed sources: key from one side via neon sign, backlights from cars, "
            "fill from street lamps"
        ),
        "shadows": "soft but multi-directional, with colored spill on faces and ground",
    },
    "soft_day_cafe": {
        "conditions": "overcast daylight spilling through large windows into a cafe interior",
        "direction": "soft key from windows, gentle bounce fill from tabletops and walls",
        "shadows": "very soft, low contrast, natural falloff",
    },
    "golden_hour_rooftop": {
        "conditions": "sunset golden hour, warm rim light and glowing sky",
        "direction": "key from low sun angle behind or 3/4 behind subjects, subtle fill from sky",
        "shadows": "long, warm-toned shadows with gentle gradients",
    },
    "blue_hour_bridge": {
        "conditions": "blue hour ambient light, city lights starting to glow",
        "direction": "cool skylight fill plus warm highlights from street lamps and car headlights",
        "shadows": "soft, cool shadows with warm edges where artificial lights hit",
    },
}

FILM_PALETTES: Dict[str, Dict[str, Any]] = {
    "kodak_portra_400": {
        "color_scheme": "warm skin tones, gentle contrast, pastel highlights, soft shadows",
        "mood_atmosphere": "nostalgic, intimate, organic music video feel",
        "contrast_hint": "medium contrast with slightly lifted blacks",
    },
    "fujifilm_eterna": {
        "color_scheme": "cool neutrals with restrained saturation, cinematic teal-orange bias",
        "mood_atmosphere": "cinematic, grounded, filmic",
        "contrast_hint": "low-to-medium contrast, smooth rolloff in highlights",
    },
    "high_contrast_bw": {
        "color_scheme": "black and white, high contrast, deep blacks, bright highlights",
        "mood_atmosphere": "dramatic, bold, graphic",
        "contrast_hint": "strong contrast with crisp edges",
    },
    "vhs_90s_mv": {
        "color_scheme": "slightly washed colors, magenta-green shifts, halation artifacts",
        "mood_atmosphere": "retro, playful, nostalgic 90s music video look",
        "contrast_hint": "low contrast with bloom around highlights",
    },
}


class FIBOBriaImageGenerator:
    """
    Thin wrapper around Bria's v2 /image/generate endpoint that speaks
    in terms of our FIBO JSON.

    We treat the FIBO JSON as the single source of truth for *what* the
    frame should look like (objects, lighting, composition, etc.) and
    only add a small layer of rendering-oriented controls such as:
      - output resolution
      - image format
      - (optional) seed
      - (optional) HDR / bit-depth hints
      - (optional) camera / lighting / film stock presets baked into FIBO

    The goal is: all of our “pro controls” live inside the FIBO JSON
    that we send as `structured_prompt`. That keeps the workflow JSON-native
    and inspectable for judges.
    """

    def __init__(
        self,
        api_key: str,
        output_dir: str = "generated",
        default_width: int = 1024,
        default_height: int = 576,
        default_image_format: str = "png",
        default_sync: bool = True,
    ) -> None:
        """
        Args:
            api_key: Bria API key (required).
            output_dir: Local directory where generated images are stored.
            default_width: Default output width in pixels (if not overridden).
            default_height: Default output height in pixels (if not overridden).
            default_image_format: "png" or "jpeg".
            default_sync: Whether to ask Bria for a synchronous response.
        """
        if not api_key:
            raise ValueError("Bria API key must not be empty")

        self.api_key = api_key
        self.base_url = "https://engine.prod.bria-api.com/v2/image/generate"

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.default_width = default_width
        self.default_height = default_height
        self.default_image_format = default_image_format
        self.default_sync = default_sync

    # -------------------------------------------------------------------------
    # Internal helpers – in-place FIBO JSON augmentation
    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_hdr_mode(
        fibo_payload: Dict[str, Any],
        hdr: bool,
        bit_depth: str,
    ) -> None:
        """
        Encode HDR / bit-depth hints both into dedicated render metadata and
        gently into the aesthetics block so the model “hears” the intent.
        """
        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["hdr"] = bool(hdr)
        render_meta["bit_depth"] = bit_depth

        aesthetics = fibo_payload.setdefault("aesthetics", {})
        existing_scheme = aesthetics.get("color_scheme", "")
        existing_mood = aesthetics.get("mood_atmosphere", "")

        if hdr or bit_depth.lower().startswith("16"):
            # Encourage rich highlight and shadow handling
            extra_scheme = "high dynamic range, smooth highlight rolloff, detailed shadows"
            extra_mood = "cinematic HDR depth and nuanced tonal separation"
        else:
            extra_scheme = "standard dynamic range"
            extra_mood = "balanced contrast"

        aesthetics["color_scheme"] = FIBOBriaImageGenerator._join_phrases(
            existing_scheme, extra_scheme
        )
        aesthetics["mood_atmosphere"] = FIBOBriaImageGenerator._join_phrases(
            existing_mood, extra_mood
        )

    @staticmethod
    def _apply_camera_preset(
        fibo_payload: Dict[str, Any],
        camera_preset: Optional[str],
    ) -> None:
        """
        Map a camera preset into the photographic_characteristics block.
        """
        if not camera_preset:
            return

        preset = CAMERA_PRESETS.get(camera_preset)
        if not preset:
            # Unknown preset – leave FIBO as-is but record for debugging.
            render_meta = fibo_payload.setdefault("_render_metadata", {})
            render_meta.setdefault("unknown_camera_presets", []).append(camera_preset)
            return

        photo = fibo_payload.setdefault("photographic_characteristics", {})
        existing_lens = photo.get("lens_focal_length", "")

        if preset.get("lens_focal_length"):
            photo["lens_focal_length"] = FIBOBriaImageGenerator._join_phrases(
                existing_lens, preset["lens_focal_length"], sep="; "
            )

        if preset.get("depth_of_field"):
            photo["depth_of_field"] = preset["depth_of_field"]

        if preset.get("shutter_speed"):
            photo["shutter_speed"] = preset["shutter_speed"]

        if preset.get("motion_blur"):
            photo["motion_blur"] = preset["motion_blur"]

        # Keep a human-readable note for debugging / UI
        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["camera_preset"] = camera_preset
        render_meta["camera_preset_notes"] = preset.get("notes", "")

    @staticmethod
    def _apply_lighting_preset(
        fibo_payload: Dict[str, Any],
        lighting_preset: Optional[str],
        scene_env: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Map a lighting blueprint into the lighting block, blending with any
        existing scene-environment cues (time_of_day / weather / setting).
        """
        if not lighting_preset:
            return

        preset = LIGHTING_BLUEPRINTS.get(lighting_preset)
        if not preset:
            render_meta = fibo_payload.setdefault("_render_metadata", {})
            render_meta.setdefault("unknown_lighting_presets", []).append(lighting_preset)
            return

        lighting = fibo_payload.setdefault("lighting", {})
        base_conditions = lighting.get("conditions", "")
        base_direction = lighting.get("direction", "")
        base_shadows = lighting.get("shadows", "")

        # Optionally fold simple scene environment hints into the conditions string
        env_suffix = ""
        if scene_env:
            env_bits = []
            time_of_day = scene_env.get("time_of_day")
            weather = scene_env.get("weather")
            setting = scene_env.get("setting")
            if time_of_day:
                env_bits.append(time_of_day)
            if weather:
                env_bits.append(weather)
            if setting:
                env_bits.append(setting)
            if env_bits:
                env_suffix = f" (scene environment: {', '.join(env_bits)})"

        conditions_text = preset.get("conditions", "") + env_suffix
        direction_text = preset.get("direction", "")
        shadows_text = preset.get("shadows", "")

        lighting["conditions"] = FIBOBriaImageGenerator._join_phrases(
            base_conditions, conditions_text
        )
        lighting["direction"] = FIBOBriaImageGenerator._join_phrases(
            base_direction, direction_text
        )
        lighting["shadows"] = FIBOBriaImageGenerator._join_phrases(
            base_shadows, shadows_text
        )

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["lighting_preset"] = lighting_preset

    @staticmethod
    def _apply_film_palette(
        fibo_payload: Dict[str, Any],
        film_palette: Optional[str],
    ) -> None:
        """
        Map film stock / palette into aesthetics (color_scheme, mood, contrast).
        """
        if not film_palette:
            return

        preset = FILM_PALETTES.get(film_palette)
        if not preset:
            render_meta = fibo_payload.setdefault("_render_metadata", {})
            render_meta.setdefault("unknown_film_palettes", []).append(film_palette)
            return

        aesthetics = fibo_payload.setdefault("aesthetics", {})
        existing_scheme = aesthetics.get("color_scheme", "")
        existing_mood = aesthetics.get("mood_atmosphere", "")

        scheme_text = preset.get("color_scheme", "")
        mood_text = preset.get("mood_atmosphere", "")
        contrast_hint = preset.get("contrast_hint", "")

        aesthetics["color_scheme"] = FIBOBriaImageGenerator._join_phrases(
            existing_scheme, scheme_text
        )
        aesthetics["mood_atmosphere"] = FIBOBriaImageGenerator._join_phrases(
            existing_mood, mood_text
        )
        if contrast_hint:
            aesthetics["contrast_behavior"] = contrast_hint

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["film_palette"] = film_palette

    @staticmethod
    def _join_phrases(
        a: str,
        b: str,
        sep: str = ", ",
    ) -> str:
        """Utility to join two descriptive phrases without duplicating separators."""
        a = (a or "").strip()
        b = (b or "").strip()
        if not a:
            return b
        if not b:
            return a
        # Avoid double commas
        if a.endswith((",", ";")):
            return f"{a} {b}"
        return f"{a}{sep}{b}"

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate_image_from_fibo_json(
        self,
        fibo_json: Dict[str, Any],
        *,
        # 1) HDR / bit-depth control
        hdr: bool = False,
        bit_depth: str = "8bit",
        # 2) Camera preset control
        camera_preset: Optional[str] = None,
        # 3) Lighting blueprint control
        lighting_preset: Optional[str] = None,
        scene_env: Optional[Dict[str, Any]] = None,
        # 4) Film stock palette
        film_palette: Optional[str] = None,
        # Existing rendering knobs
        width: Optional[int] = None,
        height: Optional[int] = None,
        image_format: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Send a FIBO JSON object as Bria's `structured_prompt` and download
        the resulting image, after applying our “pro control” layers.

        Args:
            fibo_json: The FIBO JSON payload describing the shot.
            hdr: If True, signal high dynamic range intent.
            bit_depth: One of {"8bit", "16bit"} – used in metadata & aesthetics.
            camera_preset: Optional key into CAMERA_PRESETS.
            lighting_preset: Optional key into LIGHTING_BLUEPRINTS.
            scene_env: Optional scene environment dict
                (e.g. {"time_of_day": "night", "weather": "rain", "setting": "city"}).
            film_palette: Optional key into FILM_PALETTES.
            width: Optional override for output width.
            height: Optional override for output height.
            image_format: "png" or "jpeg".
            seed: Optional random seed for reproducibility.

        Returns:
            Local file path to the downloaded image.
        """
        effective_width = width or self.default_width
        effective_height = height or self.default_height
        effective_format = image_format or self.default_image_format

        # Deep copy so we don't mutate caller state
        fibo_payload = json.loads(json.dumps(fibo_json))

        # --- Apply advanced control layers in a well-defined order ---
        self._apply_hdr_mode(fibo_payload, hdr=hdr, bit_depth=bit_depth)
        self._apply_camera_preset(fibo_payload, camera_preset=camera_preset)
        self._apply_lighting_preset(
            fibo_payload,
            lighting_preset=lighting_preset,
            scene_env=scene_env,
        )
        self._apply_film_palette(fibo_payload, film_palette=film_palette)

        # Also record target resolution and format in render metadata
        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["target_resolution"] = {
            "width": effective_width,
            "height": effective_height,
        }
        render_meta["image_format"] = effective_format

        # Build the body we send to Bria.
        payload: Dict[str, Any] = {
            "structured_prompt": json.dumps(fibo_payload),
            "sync": self.default_sync,
            "width": effective_width,
            "height": effective_height,
            "image_format": effective_format,
        }
        if seed is not None:
            payload["seed"] = seed

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

        ext = "png" if effective_format.lower().startswith("png") else "jpg"
        filename = f"fibo_{uuid.uuid4().hex}.{ext}"
        out_path = os.path.join(self.output_dir, filename)

        with open(out_path, "wb") as f:
            f.write(img_resp.content)

        return out_path