# agents/creative_director.py

from typing import Dict, List


class CreativeDirectorAgent:
    """
    Takes a script_text string and turns it into a list of shot dictionaries.

    This is a simple rule-based version so the rest of the pipeline can run:
      - /script-to-shots
      - /full-pipeline
      - /full-pipeline-fibo-json
      - /full-pipeline-generate-images

    Later, you can replace the internals with a Gemini/OpenAI LLM without
    changing the API layer.
    """

    def __init__(self) -> None:
        pass

    def script_to_shots(self, script_text: str) -> List[Dict]:
        """
        Very first stub: split the script into 1–N simple shots based
        on punctuation and keywords.

        Returns a list of dicts with the keys that the downstream agents
        expect:
          - scene (int)
          - shot_id (str)
          - description (str)
          - camera_intent (str)
          - environment (dict)
          - characters (list)
        """
        script_text = script_text.strip()
        if not script_text:
            return []

        # Naive splitting by "Scene" keyword if present, else just one scene.
        # Example: "Scene 1: ... Scene 2: ..."
        segments: List[str] = []
        current_scene = ""
        tokens = script_text.split("Scene ")
        for idx, chunk in enumerate(tokens):
            if not chunk.strip():
                continue
            if ":" in chunk:
                # e.g. "1: at night..." -> scene number + rest
                _, rest = chunk.split(":", 1)
                segments.append(rest.strip())
            else:
                # no explicit "Scene X:" pattern, treat as generic text
                segments.append(chunk.strip())

        if not segments:
            segments = [script_text]

        shots: List[Dict] = []
        shot_counter = 1

        for scene_index, seg in enumerate(segments, start=1):
            desc = seg

            lower = desc.lower()
            # Very rough camera intent guess
            if any(w in lower for w in ["establishing", "skyline", "city", "crowd"]):
                camera_intent = "wide high-angle establishing"
            elif any(w in lower for w in ["close-up", "face", "eyes", "whispers", "tear"]):
                camera_intent = "close-up eye-level"
            else:
                camera_intent = "medium eye-level"

            # Rough environment guess
            environment: Dict = {}
            if "night" in lower:
                environment["time_of_day"] = "night"
            elif "morning" in lower or "sunrise" in lower:
                environment["time_of_day"] = "morning"
            elif "evening" in lower or "sunset" in lower:
                environment["time_of_day"] = "evening"

            if "rain" in lower:
                environment["weather"] = "rain"
            elif "snow" in lower:
                environment["weather"] = "snow"

            if "city" in lower:
                environment["setting"] = "city"
            elif "alley" in lower:
                environment["setting"] = "alley"
            elif "cafe" in lower:
                environment["setting"] = "cafe"
            else:
                environment.setdefault("setting", "unspecified")

            # For now, we don't parse character names; leave empty list
            characters: List[Dict] = []

            shot = {
                "scene": scene_index,
                "shot_id": f"S{scene_index}_SHOT{shot_counter}",
                "description": desc,
                "camera_intent": camera_intent,
                "environment": environment,
                "characters": characters,
            }
            shots.append(shot)
            shot_counter += 1

        return shots