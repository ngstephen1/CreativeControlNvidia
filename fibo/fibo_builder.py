# fibo/fibo_builder.py

from typing import Any, Dict


def _build_background_setting(environment: Dict[str, Any]) -> str:
    """
    Build a simple background_setting string from environment fields.
    """
    parts = []
    setting = environment.get("setting")
    time_of_day = environment.get("time_of_day")
    weather = environment.get("weather")

    if setting:
        parts.append(setting)
    if time_of_day:
        parts.append(time_of_day)
    if weather:
        parts.append(weather)

    # e.g. "city, night, rain"
    return ", ".join(parts) if parts else "unspecified environment"


def _build_lighting_block(shot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the internal lighting representation into a FIBO-style lighting dict.
    """
    lighting_src = shot.get("lighting", {}) or {}
    environment = shot.get("environment", {}) or {}
    description = (shot.get("description") or "").lower()

    time_of_day = (environment.get("time_of_day") or "").lower()

    # Conditions description
    if "night" in time_of_day:
        conditions = (
            "nighttime environment with artificial light sources "
            "(e.g. street lamps, neon)"
        )
    elif "evening" in time_of_day or "sunset" in description:
        conditions = "warm golden-hour or sunset lighting conditions"
    else:
        conditions = "daytime environment with natural soft light"

    direction = (
        "from slightly above and in front of the subject, with subtle fill "
        "from the opposite side"
    )

    shadows = "soft, natural shadows consistent with the main light direction"

    return {
        "conditions": conditions,
        "direction": direction,
        "shadows": shadows,
    }


def _build_aesthetics_block(shot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build aesthetics (composition, color scheme, mood) from camera intent
    and environment.
    """
    camera_intent = (shot.get("camera_intent") or "").lower()
    environment = shot.get("environment", {}) or {}
    description = (shot.get("description") or "").lower()

    # Composition
    if "establishing" in camera_intent or "wide" in camera_intent:
        composition = "wide establishing shot, rule of thirds composition"
    elif "close-up" in camera_intent:
        composition = "intimate close-up framing, rule of thirds on the face"
    else:
        composition = "medium shot, balanced frame"

    # Color scheme
    time_of_day = (environment.get("time_of_day") or "").lower()
    weather = (environment.get("weather") or "").lower()
    setting = (environment.get("setting") or "").lower()

    if "night" in time_of_day:
        color_scheme = "cool blues and purples with warm highlights from artificial lights"
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
    """
    camera = shot.get("camera", {}) or {}
    aperture = float(camera.get("aperture", 2.8))

    if aperture <= 2.0:
        dof = "shallow depth of field, subject in focus with softly blurred background"
    else:
        dof = "moderate depth of field, subject and mid-ground in focus"

    camera_angle = camera.get("angle", "eye-level")
    lens_focal_length = camera.get("focal_length", "50mm")

    return {
        "depth_of_field": dof,
        "focus": "sharp focus on the main subject",
        "camera_angle": camera_angle,
        "lens_focal_length": lens_focal_length,
    }


def _build_objects(shot: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Build a minimal objects list for the FIBO JSON; for now we treat the whole
    description as a single primary subject in the center of frame.
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

    It now also includes `mv_metadata` so that music-video–aware fields
    (duration_sec, motion_style) survive all the way to downstream tools.
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

    # --- NEW: Music video metadata block ---
    fibo_json["mv_metadata"] = {
        "scene": scene_id,
        "shot_id": shot_id,
        "duration_sec": float(shot.get("duration_sec", 4.0)),
        "motion_style": shot.get(
            "motion_style",
            "slow_cinematic_push_in",
        ),
    }

    return fibo_json