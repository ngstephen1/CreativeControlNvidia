"""This module wraps Bria FIBO image generation and encodes HDR, camera, lighting,
film stock, composition, pose, mood, material, and continuity controls into the structured_prompt.
"""
import os
import uuid
import json
from typing import Any, Dict, Optional, Mapping

from pathlib import Path
from PIL import Image, ImageDraw
import requests

# If set, disables Bria API calls and returns a local stub image instead.
BRIA_OFFLINE_MODE = os.getenv("BRIA_OFFLINE_MODE", "0") == "1"


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

# ---------------------------------------------------------------------------
# NEW: Composition presets – framing & spatial design
# ---------------------------------------------------------------------------

COMPOSITION_PRESETS: Dict[str, Dict[str, Any]] = {
    "rule_of_thirds": {
        "framing_style": "rule of thirds",
        "subject_placement": "primary subject placed on thirds intersection",
        "depth_structure": "foreground and background elements supporting depth",
        "guiding_lines": "subtle leading lines into the subject",
        "camera_height": "eye-level",
        "shot_size": "medium shot",
    },
    "centered_symmetry": {
        "framing_style": "centered symmetrical composition",
        "subject_placement": "subject centered with symmetrical background",
        "depth_structure": "balanced foreground and background elements",
        "guiding_lines": "symmetry lines emphasizing stability",
        "camera_height": "eye-level",
        "shot_size": "medium to medium-close shot",
    },
    "establishing_wide": {
        "framing_style": "wide establishing shot",
        "subject_placement": "subject small relative to environment",
        "depth_structure": "deep depth with clear foreground, midground, background",
        "guiding_lines": "strong leading lines from architecture and streets",
        "camera_height": "slightly elevated",
        "shot_size": "wide or extreme wide shot",
    },
    "hero_low_angle": {
        "framing_style": "dynamic low-angle hero framing",
        "subject_placement": "subject dominant in frame, closer to camera",
        "depth_structure": "foreground textures near lens; distant background",
        "guiding_lines": "upward diagonal lines enhancing power",
        "camera_height": "low angle looking up",
        "shot_size": "medium or medium-close shot",
    },
    "intimate_closeup": {
        "framing_style": "tight, intimate framing",
        "subject_placement": "face filling much of the frame",
        "depth_structure": "very shallow depth of field, blurred background",
        "guiding_lines": "minimal lines, focus on eyes and expression",
        "camera_height": "eye-level or slightly above",
        "shot_size": "close-up or extreme close-up",
    },
}

# ---------------------------------------------------------------------------
# NEW: Pose blueprints – character acting & stance
# ---------------------------------------------------------------------------

POSE_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "performance_violinist": {
        "body_orientation": "3/4 towards camera, upper body engaged",
        "gesture": "playing violin with expressive arm movement",
        "emotion": "focused, passionate, immersed in music",
        "head_angle": "slightly down or tilted into instrument",
        "eye_focus": "eyes partially closed or looking at the violin",
        "stance": "standing, weight balanced on both feet",
        "movement": "subtle body sway following rhythm",
    },
    "reflective_window_gaze": {
        "body_orientation": "side profile near window",
        "gesture": "hands relaxed or loosely holding a cup or notebook",
        "emotion": "thoughtful, introspective",
        "head_angle": "slightly down, gaze out of frame through window",
        "eye_focus": "soft gaze into distance or reflections",
        "stance": "seated or leaning against surface",
        "movement": "minimal, still, gentle",
    },
    "confident_walk_forward": {
        "body_orientation": "facing camera, walking forward",
        "gesture": "arms relaxed, natural stride",
        "emotion": "confident, purposeful",
        "head_angle": "upright, chin slightly raised",
        "eye_focus": "looking towards or just above camera",
        "stance": "mid-step walk",
        "movement": "clear sense of motion through frame",
    },
    "crowd_supporting_subject": {
        "body_orientation": "subject facing camera, crowd behind or around",
        "gesture": "subtle gesture such as holding instrument or raising hand",
        "emotion": "uplifted, supported, connected",
        "head_angle": "slightly up, open posture",
        "eye_focus": "towards camera or slightly off",
        "stance": "standing, centered among group",
        "movement": "light movement from people around, subject more stable",
    },
}

# ---------------------------------------------------------------------------
# NEW: Mood presets – high-level tonal direction
# ---------------------------------------------------------------------------

MOOD_PRESETS: Dict[str, Dict[str, Any]] = {
    "mv_melancholy_night": {
        "tone": "melancholic, introspective, late-night city solitude",
        "energy": "low, slow visual rhythm",
        "contrast": "soft to medium contrast with gentle rolloff",
    },
    "mv_uplifting_rooftop": {
        "tone": "warm, hopeful, communal celebration",
        "energy": "medium-high, sense of connection and movement",
        "contrast": "medium contrast, glowing highlights, rich midtones",
    },
    "mv_dreamlike_cafe": {
        "tone": "dreamy, hazy, nostalgic interior calm",
        "energy": "low to medium, drifting pace",
        "contrast": "soft contrast with slight bloom in highlights",
    },
    "mv_high_energy_montage": {
        "tone": "energetic, kinetic, performance-driven",
        "energy": "high, quick visual rhythm",
        "contrast": "higher contrast, crisp edges, saturated colors",
    },
}


# ---------------------------------------------------------------------------
# NEW: Material / texture presets – surfaces & microtexture
# ---------------------------------------------------------------------------

MATERIAL_PRESETS: Dict[str, Dict[str, Any]] = {
    "neon_wet_city": {
        "texture_intent": "wet reflective streets with pronounced specular highlights",
        "specularity": "strong specular reflections on ground and metallic surfaces",
        "microtexture_behavior": "soft skin textures, crisp environmental details",
    },
    "soft_skin_portrait": {
        "texture_intent": "smooth, flattering skin with controlled pores",
        "specularity": "soft sheen, minimal harsh shine",
        "microtexture_behavior": "subtle grain, controlled fine detail on faces",
    },
    "gritty_urban": {
        "texture_intent": "gritty textures on walls, pavement, and props",
        "specularity": "mixed – matte walls with occasional wet or metal highlights",
        "microtexture_behavior": "emphasized grain and roughness in environment",
    },
    "clean_modern_minimal": {
        "texture_intent": "clean surfaces with limited texture noise",
        "specularity": "controlled highlights, smooth reflections",
        "microtexture_behavior": "low-grain, crisp edges, minimal clutter",
    },
}


def _create_placeholder_image(
    output_dir: str,
    width: int,
    height: int,
    label: str = "Bria offline / fallback",
) -> str:
    """Create a simple placeholder image when Bria is unavailable or when
    BRIA_OFFLINE_MODE is enabled. This keeps the storyboard pipeline
    running so downstream steps can still be tested.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (max(width, 64), max(height, 64)), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    text = label[:96]  # keep short to avoid wrapping issues
    draw.text((16, 16), text, fill=(230, 230, 230))
    filename = f"bria_offline_{uuid.uuid4().hex}.png"
    out_path = str(Path(output_dir) / filename)
    img.save(out_path)
    return out_path


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
      - (optional) camera / lighting / film stock presets
      - (optional) composition & pose blueprints
      - (optional) mood / intensity / material controls
      - (optional) continuity metadata

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
    # Introspection helpers – for UI / debugging
    # -------------------------------------------------------------------------

    @staticmethod
    def available_camera_presets() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the camera presets dictionary."""
        return dict(CAMERA_PRESETS)

    @staticmethod
    def available_lighting_blueprints() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the lighting blueprints dictionary."""
        return dict(LIGHTING_BLUEPRINTS)

    @staticmethod
    def available_film_palettes() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the film stock / color palette presets."""
        return dict(FILM_PALETTES)

    @staticmethod
    def available_composition_presets() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the composition presets."""
        return dict(COMPOSITION_PRESETS)

    @staticmethod
    def available_pose_blueprints() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the pose blueprints."""
        return dict(POSE_BLUEPRINTS)

    @staticmethod
    def available_mood_presets() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the high-level mood presets."""
        return dict(MOOD_PRESETS)

    @staticmethod
    def available_material_presets() -> Dict[str, Dict[str, Any]]:
        """Return a copy of the material / texture presets."""
        return dict(MATERIAL_PRESETS)

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

    # -------------------------------------------------------------------------
    # NEW: Composition + pose + mood + intensity + material + continuity
    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_composition_preset(
        fibo_payload: Dict[str, Any],
        composition_preset: Optional[str],
    ) -> None:
        """
        Map a composition preset into a dedicated 'composition' block.
        """
        if not composition_preset:
            return

        preset = COMPOSITION_PRESETS.get(composition_preset)
        if not preset:
            render_meta = fibo_payload.setdefault("_render_metadata", {})
            render_meta.setdefault("unknown_composition_presets", []).append(composition_preset)
            return

        comp = fibo_payload.setdefault("composition", {})

        def set_or_join(key: str, text: str) -> None:
            base = comp.get(key, "")
            comp[key] = FIBOBriaImageGenerator._join_phrases(base, text)

        if preset.get("framing_style"):
            set_or_join("framing_style", preset["framing_style"])
        if preset.get("subject_placement"):
            set_or_join("subject_placement", preset["subject_placement"])
        if preset.get("depth_structure"):
            set_or_join("depth_structure", preset["depth_structure"])
        if preset.get("guiding_lines"):
            set_or_join("guiding_lines", preset["guiding_lines"])
        if preset.get("camera_height"):
            set_or_join("camera_height", preset["camera_height"])
        if preset.get("shot_size"):
            set_or_join("shot_size", preset["shot_size"])

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["composition_preset"] = composition_preset

    @staticmethod
    def _apply_pose_preset(
        fibo_payload: Dict[str, Any],
        pose_preset: Optional[str],
    ) -> None:
        """
        Map a pose blueprint into 'subject_pose' block – character acting controls.
        """
        if not pose_preset:
            return

        preset = POSE_BLUEPRINTS.get(pose_preset)
        if not preset:
            render_meta = fibo_payload.setdefault("_render_metadata", {})
            render_meta.setdefault("unknown_pose_presets", []).append(pose_preset)
            return

        pose = fibo_payload.setdefault("subject_pose", {})

        def set_if_present(key: str) -> None:
            val = preset.get(key)
            if val:
                pose[key] = val

        set_if_present("body_orientation")
        set_if_present("gesture")
        set_if_present("emotion")
        set_if_present("head_angle")
        set_if_present("eye_focus")
        set_if_present("stance")
        set_if_present("movement")

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["pose_preset"] = pose_preset

    @staticmethod
    def _apply_mood_preset(
        fibo_payload: Dict[str, Any],
        mood_preset: Optional[str],
    ) -> None:
        """
        Map a high-level mood preset into aesthetics (tone, energy, contrast).
        """
        if not mood_preset:
            return

        preset = MOOD_PRESETS.get(mood_preset)
        if not preset:
            render_meta = fibo_payload.setdefault("_render_metadata", {})
            render_meta.setdefault("unknown_mood_presets", []).append(mood_preset)
            return

        aesthetics = fibo_payload.setdefault("aesthetics", {})
        existing_mood = aesthetics.get("mood_atmosphere", "")
        existing_tone = aesthetics.get("tone_description", "")

        tone_text = preset.get("tone", "")
        energy_text = preset.get("energy", "")
        contrast = preset.get("contrast", "")

        aesthetics["mood_atmosphere"] = FIBOBriaImageGenerator._join_phrases(
            existing_mood, tone_text
        )
        aesthetics["tone_description"] = FIBOBriaImageGenerator._join_phrases(
            existing_tone, energy_text
        )
        if contrast:
            aesthetics["mood_contrast_hint"] = contrast

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["mood_preset"] = mood_preset

    @staticmethod
    def _apply_intensity_controls(
        fibo_payload: Dict[str, Any],
        intensity_controls: Optional[Mapping[str, float]],
    ) -> None:
        """
        Attach a normalized 0–1 intensity control block for downstream
        tuning (sharpness, color grade, lighting, depth, etc.).
        """
        if not intensity_controls:
            return

        # Clamp to [0, 1] just in case
        clamped: Dict[str, float] = {}
        for k, v in intensity_controls.items():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f < 0.0:
                f = 0.0
            elif f > 1.0:
                f = 1.0
            clamped[k] = f

        if not clamped:
            return

        fibo_payload["intensity_controls"] = clamped

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["intensity_controls_keys"] = sorted(list(clamped.keys()))

    @staticmethod
    def _apply_material_preset(
        fibo_payload: Dict[str, Any],
        material_preset: Optional[str],
    ) -> None:
        """
        Encode material / texture behavior into a 'materials' block.
        """
        if not material_preset:
            return

        preset = MATERIAL_PRESETS.get(material_preset)
        if not preset:
            render_meta = fibo_payload.setdefault("_render_metadata", {})
            render_meta.setdefault("unknown_material_presets", []).append(material_preset)
            return

        materials = fibo_payload.setdefault("materials", {})

        def set_if_present(key: str) -> None:
            val = preset.get(key)
            if val:
                materials[key] = val

        set_if_present("texture_intent")
        set_if_present("specularity")
        set_if_present("microtexture_behavior")

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["material_preset"] = material_preset

    @staticmethod
    def _apply_continuity_tags(
        fibo_payload: Dict[str, Any],
        *,
        continuity_character_id: Optional[str],
        continuity_location_id: Optional[str],
        continuity_shot_id: Optional[str],
    ) -> None:
        """
        Store continuity IDs in a dedicated 'continuity' block so that
        downstream tools (or future models) can reason about consistency
        across shots and scenes.
        """
        if not (continuity_character_id or continuity_location_id or continuity_shot_id):
            return

        cont = fibo_payload.setdefault("continuity", {})
        if continuity_character_id:
            cont["character_id"] = continuity_character_id
        if continuity_location_id:
            cont["location_id"] = continuity_location_id
        if continuity_shot_id:
            cont["shot_id"] = continuity_shot_id

        render_meta = fibo_payload.setdefault("_render_metadata", {})
        render_meta["continuity"] = {
            "character_id": continuity_character_id,
            "location_id": continuity_location_id,
            "shot_id": continuity_shot_id,
        }

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
        # 5) Composition preset
        composition_preset: Optional[str] = None,
        # 6) Pose blueprint
        pose_preset: Optional[str] = None,
        # 7) Mood preset
        mood_preset: Optional[str] = None,
        # 8) Intensity controls (normalized 0–1)
        intensity_controls: Optional[Mapping[str, float]] = None,
        # 9) Material preset
        material_preset: Optional[str] = None,
        # 10) Continuity IDs
        continuity_character_id: Optional[str] = None,
        continuity_location_id: Optional[str] = None,
        continuity_shot_id: Optional[str] = None,
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
            composition_preset: Optional key into COMPOSITION_PRESETS.
            pose_preset: Optional key into POSE_BLUEPRINTS.
            mood_preset: Optional key into MOOD_PRESETS.
            intensity_controls: Optional mapping of normalized intensities
                (e.g. {"sharpness": 0.7, "color_grade": 0.5}).
            material_preset: Optional key into MATERIAL_PRESETS.
            continuity_character_id: Optional ID tying this shot to a character.
            continuity_location_id: Optional ID for location continuity.
            continuity_shot_id: Optional ID for shot/beat continuity.
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
        self._apply_composition_preset(fibo_payload, composition_preset=composition_preset)
        self._apply_pose_preset(fibo_payload, pose_preset=pose_preset)
        self._apply_mood_preset(fibo_payload, mood_preset=mood_preset)
        self._apply_intensity_controls(fibo_payload, intensity_controls=intensity_controls)
        self._apply_material_preset(fibo_payload, material_preset=material_preset)
        self._apply_continuity_tags(
            fibo_payload,
            continuity_character_id=continuity_character_id,
            continuity_location_id=continuity_location_id,
            continuity_shot_id=continuity_shot_id,
        )

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

        # 1) If offline mode is enabled, skip the HTTP call and emit a stub image.
        if BRIA_OFFLINE_MODE:
            return _create_placeholder_image(
                self.output_dir,
                effective_width,
                effective_height,
                label="BRIA_OFFLINE_MODE=1 (local stub)",
            )

        # 2) Call Bria image generation API with basic error handling
        try:
            resp = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=60,
            )
        except requests.RequestException as exc:
            # Network / timeout error – fallback to placeholder so the app keeps running
            return _create_placeholder_image(
                self.output_dir,
                effective_width,
                effective_height,
                label=f"Bria request error: {exc.__class__.__name__}",
            )

        if resp.status_code != 200:
            # For 5xx / gateway errors, we also fallback to a placeholder image.
            if resp.status_code >= 500:
                return _create_placeholder_image(
                    self.output_dir,
                    effective_width,
                    effective_height,
                    label=f"Bria API {resp.status_code} (fallback)",
                )
            # For 4xx errors, fail loudly so misconfiguration is visible.
            snippet = resp.text[:400]
            raise RuntimeError(f"Bria API error {resp.status_code}: {snippet}")

        try:
            data = resp.json()
        except ValueError:
            raise RuntimeError(f"Invalid JSON in Bria response: {resp.text[:400]}")

        result = data.get("result", data)
        image_url = result.get("image_url")

        if not image_url:
            raise RuntimeError(f"No image_url found in Bria response: {data}")

        # 3) Download the image to local disk
        try:
            img_resp = requests.get(image_url, timeout=60)
        except requests.RequestException as exc:
            return _create_placeholder_image(
                self.output_dir,
                effective_width,
                effective_height,
                label=f"Download error: {exc.__class__.__name__}",
            )

        if img_resp.status_code != 200:
            # Non‑200 from CDN – produce a stub so the storyboard run can continue.
            return _create_placeholder_image(
                self.output_dir,
                effective_width,
                effective_height,
                label=f"CDN error {img_resp.status_code}",
            )

        ext = "png" if effective_format.lower().startswith("png") else "jpg"
        filename = f"fibo_{uuid.uuid4().hex}.{ext}"
        out_path = os.path.join(self.output_dir, filename)

        with open(out_path, "wb") as f:
            f.write(img_resp.content)

        return out_path