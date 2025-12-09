# fibo/fibo_builder.py

"""
Utilities to convert internal shot dictionaries into FIBO-compatible
StructuredPrompt JSON objects.

This module is intentionally:
- Pure / side-effect free
- Independent of BRIA API client code

It focuses on building a *base* FIBO JSON from:
  - shot["description"]
  - shot["environment"]
  - shot["camera"]
  - shot["lighting"]

Higher-level controls such as:
  - HDR / 16-bit toggle
  - Camera lens / shutter presets
  - Lighting blueprint presets
  - Film stock color palettes

are applied later in `fibo/image_generator_bria.py`, which:
  - calls `shot_to_fibo_json(shot)` to get the base JSON
  - then layers the four control dimensions on top.
"""

from typing import Any, Dict, List


def _build_background_setting(environment: Dict[str, Any]) -> str:
    """
    Build a simple background_setting string from environment fields.

    Example:
      {"setting": "city", "time_of_day": "night", "weather": "rain"}
      -> "city, night, rain"
    """
    parts: List[str] = []
    setting = environment.get("setting")
    time_of_day = environment.get("time_of_day")
    weather = environment.get("weather")

    if setting:
        parts.append(str(setting))
    if time_of_day:
        parts.append(str(time_of_day))
    if weather:
        parts.append(str(weather))

    return ", ".join(parts) if parts else "unspecified environment"


def _build_lighting_block(shot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the internal lighting representation into a FIBO-style lighting dict.

    We intentionally keep this generic and cinematic-friendly; more specific
    lighting blueprints (e.g., "noir rim light", "soft key through window")
    are applied later by `image_generator_bria` using the lighting_preset.
    """
    lighting_src = shot.get("lighting", {}) or {}
    environment = shot.get("environment", {}) or {}
    description = (shot.get("description") or "").lower()

    time_of_day = (environment.get("time_of_day") or "").lower()

    # Conditions description
    if "night" in time_of_day:
        conditions = (
            "nighttime environment with artificial light sources "
            "(e.g. street lamps, neon, interior practicals)"
        )
    elif "evening" in time_of_day or "sunset" in description:
        conditions = "warm golden-hour or sunset lighting conditions"
    else:
        conditions = "daytime environment with natural soft light"

    # Default direction is a simple cinematic three-quarter key
    direction = (
        "from slightly above and in front of the subject, with subtle fill "
        "from the opposite side"
    )

    # Shadows stay soft and natural by default
    shadows = "soft, natural shadows consistent with the main light direction"

    # If the shot had any explicit hints (e.g. from agents), we can lightly
    # blend them in, without overfitting:
    explicit_direction = lighting_src.get("direction")
    explicit_shadows = lighting_src.get("shadows")

    if explicit_direction:
        direction = str(explicit_direction)
    if explicit_shadows:
        shadows = str(explicit_shadows)

    return {
        "conditions": conditions,
        "direction": direction,
        "shadows": shadows,
    }


def _build_aesthetics_block(shot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build aesthetics (composition, color scheme, mood) from camera intent
    and environment.

    NOTE: film stock palettes are applied later in image_generator_bria via
    `_apply_film_palette`, so this function provides a neutral cinematic base.
    """
    camera_intent = (shot.get("camera_intent") or "").lower()
    environment = shot.get("environment", {}) or {}
    description = (shot.get("description") or "").lower()

    # Composition
    if "establishing" in camera_intent or "wide" in camera_intent:
        composition = "wide establishing shot, rule of thirds composition"
    elif "close-up" in camera_intent:
        composition = "intimate close-up framing, rule of thirds on the face"
    elif "over-the-shoulder" in camera_intent:
        composition = "over-the-shoulder framing for conversational coverage"
    else:
        composition = "medium shot, balanced frame"

    # Color scheme (base, before film stock palette overrides)
    time_of_day = (environment.get("time_of_day") or "").lower()
    weather = (environment.get("weather") or "").lower()
    setting = (environment.get("setting") or "").lower()

    if "night" in time_of_day and ("city" in setting or "street" in setting):
        color_scheme = (
            "cool blues and purples with warm highlights from artificial lights"
        )
    elif "sunset" in description or "evening" in time_of_day:
        color_scheme = "warm orange and teal cinematic palette"
    elif "neon" in description or "cyberpunk" in setting:
        color_scheme = "high-saturation neon magenta and cyan accents"
    else:
        color_scheme = "soft neutral daylight colors"

    # Mood
    if any(w in description for w in ["lonely", "melancholic", "sad"]):
        mood = "melancholic, introspective atmosphere"
    elif any(w in description for w in ["hopeful", "uplifting", "joyful"]):
        mood = "hopeful, uplifting atmosphere"
    elif "tense" in description or "suspense" in description:
        mood = "tense, suspenseful atmosphere"
    else:
        mood = "cinematic, contemplative atmosphere"

    return {
        "composition": composition,
        "color_scheme": color_scheme,
        "mood_atmosphere": mood,
    }


def _build_photographic_characteristics(shot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the photographic_characteristics block from the camera dict.

    Camera-level stylistic overrides (HDR, film stock, etc.) are layered later,
    but we encode a sensible base representation here.
    """
    camera = shot.get("camera", {}) or {}

    aperture_raw = camera.get("aperture", 2.8)
    try:
        aperture = float(aperture_raw)
    except (TypeError, ValueError):
        aperture = 2.8

    if aperture <= 2.0:
        dof = "shallow depth of field, subject in focus with softly blurred background"
    elif aperture >= 8.0:
        dof = "deep depth of field, most of the scene in focus"
    else:
        dof = "moderate depth of field, subject and mid-ground in focus"

    camera_angle = camera.get("angle", "eye-level")
    lens_focal_length = camera.get("focal_length", "50mm")

    # Some cameras may supply additional hints:
    shutter_speed = camera.get("shutter_speed")  # e.g. "1/48s"
    sensor_format = camera.get("sensor_format")  # e.g. "Super35", "full-frame"

    characteristics: Dict[str, Any] = {
        "depth_of_field": dof,
        "focus": "sharp focus on the main subject",
        "camera_angle": camera_angle,
        "lens_focal_length": lens_focal_length,
    }

    # Only include extra technical fields if present; FIBO is tolerant to
    # additional keys, and these can be useful for debugging / future models.
    if shutter_speed:
        characteristics["shutter_speed"] = str(shutter_speed)
    if sensor_format:
        characteristics["sensor_format"] = str(sensor_format)

    return characteristics


def _build_objects(shot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a minimal objects list for the FIBO JSON; for now we treat the whole
    description as a single primary subject in the center of frame.

    Later, more advanced object breakdowns (characters, props, set dressing)
    can be added by higher-level agents.
    """
    description = shot.get("description", "")
    if not description:
        description = "Main subject of the shot"

    return [
        {
            "description": description,
            "location": "center of frame",
            "relationship": "primary_subject",
            "relative_size": "medium",
            "shape_and_color": "",
            "texture": "",
            "appearance_details": "",
            "number_of_objects": 1,
            "pose": "",
            "expression": "",
            "clothing": "",
            "action": "",
            "gender": "",
            "skin_tone_and_texture": "",
            "orientation": "facing camera",
        }
    ]


def shot_to_fibo_json(shot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an internal shot dict into a FIBO-compatible StructuredPrompt JSON.

    This is the main bridge between:
      - CreativeDirector / Cinematography / Continuity agents
      - Bria FIBO JSON-native generation

    It also includes an `mv_metadata` block so that music-video–aware fields
    (duration_sec, motion_style, camera / lighting / palette presets, hdr_16bit)
    survive all the way to downstream tools.
    """
    description = shot.get("description") or ""
    scene_id = shot.get("scene")
    shot_id = shot.get("shot_id")

    environment = shot.get("environment", {}) or {}

    short_description = description.strip() or "Storyboard frame"

    background_setting = _build_background_setting(environment)
    lighting_block = _build_lighting_block(shot)
    aesthetics_block = _build_aesthetics_block(shot)
    photo_char_block = _build_photographic_characteristics(shot)
    objects_block = _build_objects(shot)

    # Base FIBO JSON structure
    fibo_json: Dict[str, Any] = {
        "short_description": short_description,
        "objects": objects_block,
        "background_setting": background_setting,
        "lighting": lighting_block,
        "aesthetics": aesthetics_block,
        "photographic_characteristics": photo_char_block,
        "style_medium": "photograph, cinematic storyboard frame",
        "text_render": [],
        "context": (
            f"Storyboard frame generated by Autonomous Studio Director "
            f"for scene {scene_id}, shot {shot_id}."
        ),
        "artistic_style": "cinematic realistic, detailed, filmic look",
    }

    # --- Music video + control metadata block ---
    fibo_json["mv_metadata"] = {
        "scene": scene_id,
        "shot_id": shot_id,
        "duration_sec": float(shot.get("duration_sec", 4.0)),
        "motion_style": shot.get(
            "motion_style",
            "slow_cinematic_push_in",
        ),
        # Control presets (read by image_generator_bria + video backend)
        "camera_preset": shot.get("camera_preset"),
        "lighting_preset": shot.get("lighting_preset"),
        "film_palette": shot.get("film_palette"),
        "hdr_16bit": bool(shot.get("hdr_16bit", False)),
        # Additional creative controls (kept even if None, so downstream code
        # can always rely on the keys being present)
        "composition_preset": shot.get("composition_preset"),
        "pose_preset": shot.get("pose_preset"),
        "mood_preset": shot.get("mood_preset"),
        "mood_intensity": shot.get("mood_intensity"),
        "material_preset": shot.get("material_preset"),
        # Generic continuity dictionary (e.g. character/location/prop IDs)
        "continuity": shot.get("continuity", {}),
    }

    return fibo_json