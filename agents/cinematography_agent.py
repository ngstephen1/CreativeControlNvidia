# agents/cinematography_agent.py

from typing import Dict, List


class CinematographyAgent:
    """
    Takes shot-level JSON from CreativeDirectorAgent and adds concrete
    camera + lighting parameters.

    For now this is rule-based (no LLM), so it runs fast on CPU.

    It is also aware of basic music video semantics:
      - If the shot already has a `motion_style` from the Creative Director,
        we may refine it based on camera intent and environment.
      - If it does NOT have `motion_style`, we will set a sensible default.
    """

    def __init__(self) -> None:
        # You can store global defaults or presets here if needed
        self.default_iso = 400

    def _infer_motion_style(self, shot: Dict) -> str:
        """
        Infer or refine a motion_style hint for downstream image→video backends
        based on camera intent, camera angle, and environment.

        Examples:
          - establishing / wide city → slow_dolly_out
          - intimate close-up → handheld_drift
          - performance / stage → orbital_around_subject
        """
        description = shot.get("description", "").lower()
        camera_intent = shot.get("camera_intent", "").lower()
        environment = shot.get("environment", {})
        setting = str(environment.get("setting", "")).lower()

        # If the Creative Director has already suggested something, take it
        # as the starting point.
        base_motion = shot.get("motion_style", "").strip()

        # Strong, specific overrides first
        if "stage" in setting or "concert" in setting:
            return "orbital_around_subject"
        if "crowd" in description or "dance" in description:
            return "handheld_drift"

        # Camera-intent based mapping
        if "establishing" in camera_intent or "wide" in camera_intent:
            return "slow_dolly_out"
        if "close-up" in camera_intent or "closeup" in camera_intent:
            return "handheld_drift"
        if "over-the-shoulder" in camera_intent:
            return "handheld_drift"

        # Fallback if we already had something reasonable
        if base_motion:
            return base_motion

        # Safe default
        return "slow_cinematic_push_in"

    def enrich_shot(self, shot: Dict) -> Dict:
        """
        Given a single shot dict, add 'camera' and 'lighting' fields
        based on the shot description and camera_intent, and refine
        `motion_style` for music-video–aware image→video rendering.
        """
        description = shot.get("description", "").lower()
        camera_intent = shot.get("camera_intent", "").lower()

        # --- Camera presets ---
        camera: Dict = {}

        # Focal length & FOV based on rough shot type
        if "establishing" in description or "wide" in camera_intent:
            camera["focal_length"] = "24mm"
            camera["fov"] = 80
            camera["distance_to_subject"] = "far"
        elif "close" in description or "close-up" in camera_intent:
            camera["focal_length"] = "85mm"
            camera["fov"] = 30
            camera["distance_to_subject"] = "near"
        else:
            # default medium shot
            camera["focal_length"] = "50mm"
            camera["fov"] = 50
            camera["distance_to_subject"] = "medium"

        # Camera angle from intent
        if "low-angle" in camera_intent or "low angle" in camera_intent:
            camera["angle"] = "low"
        elif "high-angle" in camera_intent or "high angle" in camera_intent:
            camera["angle"] = "high"
        else:
            camera["angle"] = "eye-level"

        # Aperture / exposure based on mood
        if any(word in description for word in ["dramatic", "moody", "night"]):
            camera["aperture"] = 1.8
            camera["exposure_compensation"] = -0.7
            camera["iso"] = 800
        else:
            camera["aperture"] = 2.8
            camera["exposure_compensation"] = 0.0
            camera["iso"] = self.default_iso

        # --- Lighting presets ---
        environment = shot.get("environment", {})
        time_of_day = environment.get("time_of_day", "").lower()
        lighting: Dict = {}

        if "night" in time_of_day:
            lighting["key_light_temperature"] = 4500  # cooler
            lighting["fill_light_temperature"] = 3800
            lighting["contrast_ratio"] = "high"
        elif "sunset" in description:
            lighting["key_light_temperature"] = 6500  # warm
            lighting["fill_light_temperature"] = 5500
            lighting["contrast_ratio"] = "medium"
        else:
            # generic daytime
            lighting["key_light_temperature"] = 5600
            lighting["fill_light_temperature"] = 5000
            lighting["contrast_ratio"] = "low"

        # Neon mood based on environment
        setting = environment.get("setting", "").lower()
        if "neon" in description or "cyberpunk" in setting:
            lighting["accent_lights"] = ["neon magenta", "neon cyan"]
        else:
            lighting["accent_lights"] = []

        # Attach to shot
        shot["camera"] = camera
        shot["lighting"] = lighting

        # --- Music-video motion refinement ---
        shot["motion_style"] = self._infer_motion_style(shot)

        return shot

    def enrich_shots(self, shots: List[Dict]) -> List[Dict]:
        """
        Apply enrich_shot to a list of shots.
        """
        return [self.enrich_shot(shot.copy()) for shot in shots]