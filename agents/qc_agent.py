# agents/qc_agent.py

from typing import Dict, List, Any


class QualityControlAgent:
    """
    Runs simple checks on shots JSON to catch:
    - missing fields
    - obvious contradictions
    - under-specified shots
    """

    def __init__(self):
        pass

    def check_shot(self, shot: Dict[str, Any]) -> List[str]:
        issues: List[str] = []

        # basic keys
        if "camera" not in shot:
            issues.append("Missing 'camera' block.")
        if "lighting" not in shot:
            issues.append("Missing 'lighting' block.")
        if "environment" not in shot:
            issues.append("Missing 'environment' block.")

        # time_of_day vs description mismatch
        env = shot.get("environment", {})
        time_of_day = (env.get("time_of_day") or "").lower()
        desc = (shot.get("description") or "").lower()

        if "night" in desc and time_of_day not in ["night", ""]:
            issues.append(
                f"Description says 'night' but environment.time_of_day='{time_of_day}'."
            )

        if "sunset" in desc and time_of_day not in ["sunset", "evening", ""]:
            issues.append(
                f"Description mentions 'sunset' but time_of_day='{time_of_day}'."
            )

        # camera plausibility checks
        camera = shot.get("camera", {})
        fov = camera.get("fov")
        focal = camera.get("focal_length")

        if fov is not None and isinstance(fov, (int, float)):
            if fov < 20 or fov > 120:
                issues.append(f"FOV={fov} looks unusual (expected roughly 20–120 degrees).")

        # minimal characters info
        chars = shot.get("characters", [])
        for ch in chars:
            if "name" not in ch:
                issues.append("Character without a 'name' field.")
            if "face_seed" not in ch:
                issues.append(f"Character '{ch.get('name', '?')}' missing face_seed (continuity).")

        return issues

    def check_shots(self, shots: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Returns a dict mapping shot_id to list of issues.
        """
        report: Dict[str, List[str]] = {}
        for shot in shots:
            shot_id = shot.get("shot_id", "UNKNOWN")
            issues = self.check_shot(shot)
            if issues:
                report[shot_id] = issues
        return report