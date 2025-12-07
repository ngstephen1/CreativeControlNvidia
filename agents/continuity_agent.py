# agents/continuity_agent.py

from typing import Dict, List
import hashlib
import random


class ContinuityAgent:
    """
    Ensures consistency across shots:
    - character identity (face_seed, appearance)
    - shared environment settings per scene
    """

    def __init__(self):
        # store continuity bible in-memory for now
        self.character_bible: Dict[str, Dict] = {}
        self.scene_environment: Dict[int, Dict] = {}

    # ---------- helpers ----------

    def _get_deterministic_seed(self, key: str) -> int:
        """
        Create a deterministic seed from a string key.
        """
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # take first 8 hex chars -> int
        return int(h[:8], 16)

    def _init_character_if_needed(self, character: Dict) -> Dict:
        """
        Ensure a character entry has a consistent record in character_bible.
        """
        name = character.get("name", "Unknown")
        if name not in self.character_bible:
            seed = self._get_deterministic_seed(name)
            rnd = random.Random(seed)

            # basic appearance defaults for now
            self.character_bible[name] = {
                "name": name,
                "face_seed": seed,
                "hair_color": rnd.choice(["black", "brown", "blonde", "red"]),
                "eye_color": rnd.choice(["brown", "blue", "green", "hazel"]),
                "base_outfit": rnd.choice(
                    ["coat and hat", "suit", "hoodie and jeans", "casual shirt"]
                ),
            }

        # merge character info with bible (bible is canonical)
        merged = {**self.character_bible[name], **character}
        return merged

    def _init_scene_env_if_needed(self, scene_id: int, environment: Dict) -> Dict:
        """
        Ensure a scene environment entry exists and merge consistent fields.
        """
        if scene_id not in self.scene_environment:
            # first time we see this scene, store environment
            self.scene_environment[scene_id] = environment.copy()
            return self.scene_environment[scene_id]

        # merge: bible takes precedence if conflict
        existing = self.scene_environment[scene_id]
        merged = {**existing, **environment}
        self.scene_environment[scene_id] = merged
        return merged

    # ---------- public API ----------

    def apply_continuity(self, shots: List[Dict]) -> List[Dict]:
        """
        Takes a list of shot dicts and enforces:
        - consistent characters
        - consistent scene-level environment
        """
        output = []

        for shot in shots:
            scene_id = shot.get("scene", 1)
            environment = shot.get("environment", {})
            characters = shot.get("characters", [])

            # scene environment continuity
            env_consistent = self._init_scene_env_if_needed(scene_id, environment)
            shot["environment"] = env_consistent

            # character continuity
            consistent_characters = []
            for ch in characters:
                consistent_characters.append(self._init_character_if_needed(ch))
            shot["characters"] = consistent_characters

            output.append(shot)

        return output

    def get_character_bible(self) -> Dict[str, Dict]:
        return self.character_bible

    def get_scene_environment_bible(self) -> Dict[int, Dict]:
        return self.scene_environment