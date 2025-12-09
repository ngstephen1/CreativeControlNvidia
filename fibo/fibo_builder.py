"""fibo/fibo_builder.py

Utilities to convert internal shot dictionaries into FIBO-compatible
StructuredPrompt JSON objects and to derive secondary representations like
DSL summaries and ComfyUI export templates.

Design principles
-----------------
- Pure / side-effect free
- Independent of BRIA API client code
- Focused on shaping Python dicts (no network / disk I/O)

The main responsibilities here are:
  1. Build a *base* FIBO JSON from an internal `shot` dict via
     `shot_to_fibo_json(shot)`.
  2. Provide a compact, human-readable DSL summary string for continuity
     and debugging via `shot_to_dsl_summary(...)`.
  3. Build a ComfyUI-ready JSON *template* that encodes each FIBO JSON as
     a node input via `build_comfyui_template(...)`.

Higher-level controls such as:
  - HDR / 16-bit toggle
  - Camera lens / shutter presets
  - Lighting blueprint presets
  - Film stock color palettes
  - Composition / pose / material presets

are applied later in `fibo/image_generator_bria.py`, which:
  - calls `shot_to_fibo_json(shot)` to get the base JSON
  - then layers the control dimensions on top.
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Low-level FIBO JSON builders
# ---------------------------------------------------------------------------


def _build_background_setting(environment: Dict[str, Any]) -> str:
    """Build a simple background_setting string from environment fields.

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
    """Convert the internal lighting representation into a FIBO-style
    lighting dict.

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
    """Build aesthetics (composition, color scheme, mood) from camera intent
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
    """Build the photographic_characteristics block from the camera dict.

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
        dof = (
            "shallow depth of field, subject in focus with softly blurred "
            "background"
        )
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
    """Build a minimal objects list for the FIBO JSON; for now we treat the
    whole description as a single primary subject in the center of frame.

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
    """Convert an internal shot dict into a FIBO-compatible StructuredPrompt
    JSON.

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
            "Storyboard frame generated by Autonomous Studio Director "
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


# ---------------------------------------------------------------------------
# Secondary representations: DSL summary & ComfyUI template
# ---------------------------------------------------------------------------


def _safe_get_mv_field(
    shot: Dict[str, Any] | None,
    fibo_json: Dict[str, Any] | None,
    key: str,
    default: Any | None = None,
) -> Any | None:
    """Helper to pull a field from mv_metadata first, then from the raw shot.

    This lets downstream code remain agnostic about whether it is working
    directly from FIBO JSON or from the original shot dict.
    """
    if fibo_json is not None:
        mv_meta = fibo_json.get("mv_metadata") or {}
        if key in mv_meta and mv_meta[key] is not None:
            return mv_meta[key]

    if shot is not None and key in shot and shot[key] is not None:
        return shot[key]

    return default


def shot_to_dsl_summary(shot: Dict[str, Any], fibo_json: Dict[str, Any] | None = None) -> str:
    """Build a compact, human-readable DSL-style summary for a shot.

    This is used by the Continuity Inspector and any UI elements that need
    a single-line description of the shot's technical look, e.g.:

      "S1–SHOT3 | CAM: low-angle, 35mm | LIGHT: warm café window light |\n"
      "LOOK: teal-orange / melancholic, introspective atmosphere | HDR"

    The function is resilient to missing keys and will gracefully fall back
    to defaults where necessary.
    """
    fibo_json = fibo_json or {}

    mv_meta = fibo_json.get("mv_metadata") or {}
    aesthetics = fibo_json.get("aesthetics") or {}
    lighting = fibo_json.get("lighting") or {}
    photo = fibo_json.get("photographic_characteristics") or {}

    scene = _safe_get_mv_field(shot, fibo_json, "scene")
    shot_id = _safe_get_mv_field(shot, fibo_json, "shot_id")

    # Camera
    camera_angle = photo.get("camera_angle") or _safe_get_mv_field(
        shot, fibo_json, "camera_angle", "eye-level"
    )
    lens_focal_length = photo.get("lens_focal_length") or _safe_get_mv_field(
        shot, fibo_json, "lens_focal_length", "50mm"
    )

    # Lighting / palette
    lighting_preset = mv_meta.get("lighting_preset")
    film_palette = mv_meta.get("film_palette")
    hdr_flag = bool(mv_meta.get("hdr_16bit", False))

    color_scheme = aesthetics.get("color_scheme")
    mood = aesthetics.get("mood_atmosphere") or mv_meta.get("mood_preset")

    # Build the pieces
    parts: List[str] = []

    if scene is not None and shot_id is not None:
        parts.append(f"S{scene}–{shot_id}")

    cam_desc = f"{camera_angle}, {lens_focal_length}".strip(", ")
    if cam_desc:
        parts.append(f"CAM: {cam_desc}")

    light_desc = lighting_preset or lighting.get("conditions")
    if light_desc:
        parts.append(f"LIGHT: {light_desc}")

    look_bits: List[str] = []
    if film_palette:
        look_bits.append(str(film_palette))
    if color_scheme:
        look_bits.append(str(color_scheme))
    if mood:
        look_bits.append(str(mood))
    if look_bits:
        parts.append("LOOK: " + " / ".join(look_bits))

    parts.append("HDR" if hdr_flag else "SDR")

    return " | ".join(parts)


def build_comfyui_template(
    shots: List[Dict[str, Any]],
    fibo_payloads: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a *ComfyUI-style* graph template for a sequence of shots.

    This does **not** try to mirror the exact internal node schema of
    ComfyUI-BRIA. Instead, it provides a stable, declarative JSON that a
    ComfyUI workflow (or custom node) can ingest and map into the actual
    graph.

    The shape is intentionally simple:

    {
      "template_type": "comfyui-graph",
      "version": "0.1.0",
      "description": "Autogenerated BRIA FIBO graph for music-video storyboard shots.",
      "nodes": {
        "shot_00": {
          "class_type": "BRIA_FIBO_Generate",
          "inputs": {
            "json_prompt": { ... full FIBO JSON ... },
            "seed": 123,
            "cfg_scale": 5.0,
            "steps": 25,
            "width": 1024,
            "height": 1024
          },
          "_meta": {
            "scene": 1,
            "shot_id": "SHOT1",
            "dsl_summary": "S1–SHOT1 | CAM: ...",
          }
        },
        ...
      },
      "meta": {
        "num_shots": 7
      }
    }

    A downstream ComfyUI loader can:
      - Iterate over `nodes` in order.
      - For each node, read `inputs["json_prompt"]` and wire it into an
        actual BRIA-FIBO generation node.
      - Use `_meta` to display friendly labels in the UI.
    """
    if len(shots) != len(fibo_payloads):
        raise ValueError(
            "build_comfyui_template: shots and fibo_payloads must be same length"
        )

    nodes: Dict[str, Any] = {}

    for idx, (shot, fibo_json) in enumerate(zip(shots, fibo_payloads)):
        node_id = f"shot_{idx:02d}"

        mv_meta = fibo_json.get("mv_metadata") or {}

        # Basic tech params with gentle defaults
        width = int(shot.get("width") or mv_meta.get("width") or 1024)
        height = int(shot.get("height") or mv_meta.get("height") or 1024)
        steps = int(shot.get("steps") or mv_meta.get("steps") or 25)
        cfg_scale = float(shot.get("cfg_scale") or mv_meta.get("cfg_scale") or 5.0)
        seed_raw = shot.get("seed") or mv_meta.get("seed") or idx
        try:
            seed = int(seed_raw)
        except (TypeError, ValueError):
            seed = idx

        node: Dict[str, Any] = {
            "class_type": "BRIA_FIBO_Generate",
            "inputs": {
                "json_prompt": fibo_json,
                "seed": seed,
                "cfg_scale": cfg_scale,
                "steps": steps,
                "width": width,
                "height": height,
            },
            "_meta": {
                "scene": mv_meta.get("scene"),
                "shot_id": mv_meta.get("shot_id"),
                "dsl_summary": shot_to_dsl_summary(shot, fibo_json),
            },
        }

        nodes[node_id] = node

    template: Dict[str, Any] = {
        "template_type": "comfyui-graph",
        "version": "0.1.0",
        "description": "Autogenerated BRIA FIBO graph for music-video storyboard shots.",
        "nodes": nodes,
        "meta": {
            "num_shots": len(shots),
        },
    }

    return template