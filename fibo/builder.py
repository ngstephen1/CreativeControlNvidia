# fibo/builder.py

from typing import Any, Dict, List


class FIBOSceneBuilder:
    """
    Build a FIBO-compatible StructuredPrompt JSON from an internal 'shot' dict.

    Target schema (from Bria/FIBO docs):

      StructuredPrompt {
        short_description: string
        objects: list<PromptObject>
        background_setting: string
        lighting: {
          conditions: string
          direction: string
          shadows: string
        }
        aesthetics: {
          composition: string
          color_scheme: string
          mood_atmosphere: string
        }
        photographic_characteristics: {
          depth_of_field: string
          focus: string
          camera_angle: string
          lens_focal_length: string
        }
        style_medium: string
        text_render: list
        context: string
        artistic_style: string
      }
    """

    def __init__(self) -> None:
        pass

    def shot_to_fibo_json(self, shot: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = shot.get("scene", 1)
        shot_id = shot.get("shot_id", "")
        description: str = (
            shot.get("description")
            or shot.get("description_text")
            or shot.get("camera_intent")
            or ""
        )

        environment: Dict[str, Any] = shot.get("environment") or {}
        camera: Dict[str, Any] = shot.get("camera") or {}
        lighting_src: Dict[str, Any] = shot.get("lighting") or {}
        mood: str = (
            shot.get("mood")
            or environment.get("mood", "")
        )

        # -----------------------------
        # background_setting string
        # -----------------------------
        bg_parts: List[str] = []

        setting = environment.get("setting")
        if setting:
            bg_parts.append(str(setting))

        time_of_day = environment.get("time_of_day")
        if time_of_day:
            bg_parts.append(str(time_of_day))

        weather = environment.get("weather") or environment.get("weather_type")
        if weather:
            bg_parts.append(str(weather))

        background_setting = ", ".join(bg_parts) or "unspecified environment"

        # -----------------------------
        # objects: simple single main subject for now
        # -----------------------------
        objects: List[Dict[str, Any]] = []

        main_obj_desc = description or "main subject of the shot"
        objects.append(
            {
                "description": main_obj_desc,
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
        )

        # -----------------------------
        # lighting
        # -----------------------------
        tod_lower = (time_of_day or "").lower()
        if tod_lower == "night":
            conditions = "nighttime environment with artificial light sources (e.g. street lamps, neon)"
        elif tod_lower == "morning":
            conditions = "soft morning daylight"
        elif tod_lower == "evening":
            conditions = "warm evening light around sunset"
        else:
            conditions = lighting_src.get(
                "conditions", "soft cinematic lighting with gentle contrast"
            )

        direction = lighting_src.get(
            "direction",
            "from slightly above and in front of the subject, with subtle fill from the opposite side",
        )
        shadows = lighting_src.get(
            "shadows",
            "soft, natural shadows consistent with the main light direction",
        )

        lighting = {
            "conditions": conditions,
            "direction": direction,
            "shadows": shadows,
        }

        # -----------------------------
        # aesthetics
        # -----------------------------
        camera_intent = shot.get("camera_intent", "").lower()
        if "wide" in camera_intent:
            composition = "wide establishing shot, rule of thirds composition"
        elif "close" in camera_intent:
            composition = "tight framing on the main subject"
        else:
            composition = "balanced, medium framing on the main subject"

        color_scheme = (
            environment.get("color_scheme")
            or lighting_src.get("color_scheme")
        )
        if not color_scheme:
            if tod_lower == "night":
                color_scheme = "cool blues and purples with warm highlights from artificial lights"
            else:
                color_scheme = "neutral cinematic palette with soft contrast"

        mood_atmosphere = (
            mood
            or environment.get("mood")
            or "cinematic, contemplative atmosphere"
        )

        aesthetics = {
            "composition": composition,
            "color_scheme": color_scheme,
            "mood_atmosphere": mood_atmosphere,
        }

        # -----------------------------
        # photographic_characteristics
        # -----------------------------
        photographic_characteristics = {
            "depth_of_field": "shallow depth of field, subject in focus with softly blurred background",
            "focus": "sharp focus on the main subject",
            "camera_angle": camera.get("angle", "eye-level"),
            "lens_focal_length": camera.get("focal_length", "50mm"),
        }

        # -----------------------------
        # structured prompt (FIBO schema)
        # -----------------------------
        short_description = (
            description
            or f"Storyboard frame for scene {scene_id}, shot {shot_id}"
        )

        structured_prompt: Dict[str, Any] = {
            "short_description": short_description,
            "objects": objects,
            "background_setting": background_setting,
            "lighting": lighting,
            "aesthetics": aesthetics,
            "photographic_characteristics": photographic_characteristics,
            "style_medium": "photograph, cinematic storyboard frame",
            "text_render": [],
            "context": (
                f"Storyboard frame generated by Autonomous Studio Director "
                f"for scene {scene_id}, shot {shot_id}."
            ),
            "artistic_style": "cinematic realistic, detailed, filmic look",
        }

        return structured_prompt