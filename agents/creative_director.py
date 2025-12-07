# agents/creative_director.py

from typing import List, Dict

class CreativeDirectorAgent:
    """
    Takes raw script text and produces a list of shot-level JSON templates.
    """

    def __init__(self):
        pass  # later: inject LLM client here

    def script_to_shots(self, script_text: str) -> List[Dict]:
        """
        Very first stub: returns a dummy shot breakdown.
        Later: call LLM to parse full scenes.
        """
        # TODO: replace with real logic
        return [
            {
                "scene": 1,
                "shot_id": "1A",
                "description": "Establishing shot of city at night.",
                "camera_intent": "wide, high-angle",
                "environment": {
                    "setting": "city",
                    "time_of_day": "night"
                },
                "characters": []
            }
        ]