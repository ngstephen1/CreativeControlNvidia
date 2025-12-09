"""
gemini/character_annotator.py

Use Gemini 2.5 Pro to extract advanced character–continuity annotations
from a single frame. The result is mapped into our Pydantic
`CharacterAnnotation` model so it can be surfaced in the Continuity
Inspector UI.

This module is intentionally self‑contained and defensive:
- clear error if GEMINI_API_KEY is missing
- robust JSON parsing with fallbacks
- optional scene / shot context to guide the model
- helper for batch annotation
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ValidationError
from PIL import Image

import google.generativeai as genai  # pip install google-generativeai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local Pydantic models (self-contained, avoid circular imports with app.api)
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):
    """Simple pixel-space bounding box for the main character's face."""

    x: int
    y: int
    width: int
    height: int


class CharacterAnnotation(BaseModel):
    """
    Structured description of a single main character in a frame,
    focused on continuity-critical attributes.
    """

    shot_id: Optional[str] = None
    scene_prompt: Optional[str] = None

    # Visual identity
    character_name: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None  # face or head region
    hair_color: Optional[str] = None

    # Wardrobe & props
    clothing: List[str] = []
    props: List[str] = []

    # Expression / pose
    expression: Optional[str] = None
    pose: Optional[str] = None

    # Short tags we want to keep consistent across shots
    continuity_tags: List[str] = []

    # Free-form notes from the model
    raw_model_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Set it in your environment (or Streamlit secrets) before using the "
        "character annotator."
    )

genai.configure(api_key=GEMINI_API_KEY)

# Use the latest, strongest multimodal model
MODEL_NAME = "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# Prompt + JSON schema
# ---------------------------------------------------------------------------

JSON_SCHEMA = """
{
  "shot_id": "string | null",
  "bounding_box": {
    "x": "int (left pixel, 0-based)",
    "y": "int (top pixel, 0-based)",
    "width": "int (pixels)",
    "height": "int (pixels)"
  } | null,
  "hair_color": "short phrase, e.g. 'black', 'dark brown with blue tint'",
  "clothing": ["coat", "scarf", "shirt", "..."],
  "props": ["objects clearly held or worn, e.g. 'violin', 'umbrella'"],
  "expression": "short phrase, e.g. 'calm, slight smile'",
  "pose": "short phrase describing body/pose, e.g. 'facing camera, shoulders turned left'",
  "continuity_tags": [
    "very short phrases for details that must stay consistent across shots, e.g.",
    "'hair tucked behind ears', 'grey knit scarf', 'navy wool coat'"
  ],
  "raw_model_notes": "free-form notes for the storyboard team"
}
""".strip()

BASE_SYSTEM_PROMPT = f"""
You are a senior continuity supervisor for a film shoot.

Your task:
Given a SINGLE frame (image), describe the MAIN HUMAN CHARACTER ONLY
in terms of appearance and continuity-critical details.

Important:
- If multiple people appear, choose the one that is most centered or
  most visually dominant (largest in frame).
- Ignore background extras.

You MUST return STRICT JSON that matches this exact schema:

{JSON_SCHEMA}

Guidelines:
- Be concise but specific.
- The bounding_box is an approximate rectangle around the main character's face.
- clothing: only key visible pieces (coat, scarf, dress, shirt, pants, etc.).
- props: only clearly visible objects the character is holding or wearing
  (e.g. violin, phone, bag).
- continuity_tags: the most important visual details that should remain
  consistent across shots (hair style, color, key garments, distinctive props).
- raw_model_notes: any additional notes for the storyboard / QA team.

Return ONLY valid JSON. Do not include explanations, markdown, or text
outside the JSON object.
""".strip()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_path(p: Union[str, Path]) -> Path:
    """Convert string or Path to a Path instance."""
    return p if isinstance(p, Path) else Path(p)


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to open image {path}: {exc}") from exc


def _build_prompt(scene_prompt: Optional[str] = None, shot_id: Optional[str] = None) -> str:
    """Add optional scene / shot context on top of the base system prompt."""
    extras: List[str] = []
    if scene_prompt:
        extras.append(f"Scene description: {scene_prompt}")
    if shot_id:
        extras.append(f"Shot identifier: {shot_id}")
    if not extras:
        return BASE_SYSTEM_PROMPT
    return BASE_SYSTEM_PROMPT + "\n\nContext:\n" + "\n".join(extras)


def _parse_gemini_json(resp: Any) -> Dict[str, Any]:
    """
    Robustly parse JSON output from Gemini.

    We request `response_mime_type='application/json'`, but still defend
    against:
    - `resp.text` containing JSON
    - minor trailing text / markdown
    """
    text = ""
    try:
        # Modern google-generativeai responses expose `.text`
        text = getattr(resp, "text", "") or ""
    except Exception:  # pragma: no cover - best-effort
        text = ""

    if not text:
        # Fallback: some SDK versions expose candidates[0].content.parts[0].text
        try:  # pragma: no cover - defensive fallback
            candidate = resp.candidates[0]
            parts = candidate.content.parts
            text = "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            pass

    if not text:
        raise RuntimeError("Gemini returned empty response; cannot parse JSON.")

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: attempt to extract JSON between first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse JSON snippet from Gemini: %s", exc)
        raise RuntimeError(f"Failed to parse JSON from Gemini response: {text}")


def _ensure_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure optional list fields exist so Pydantic doesn't get `None`."""
    data = dict(data)  # shallow copy
    data.setdefault("clothing", [])
    data.setdefault("props", [])
    data.setdefault("continuity_tags", [])
    return data


def _normalize_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize key naming differences between the prompt schema
    and whatever Gemini returns.

    For example, if Gemini returns `face_box` instead of `bounding_box`,
    we map it.
    """
    if "bounding_box" not in data and "face_box" in data:
        data["bounding_box"] = data.pop("face_box")
    return data


def _get_model() -> genai.GenerativeModel:
    """Factory for the Gemini model (helps if we ever swap configs)."""
    return genai.GenerativeModel(MODEL_NAME)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_character_image(
    image_path: Union[str, Path],
    shot_id: Optional[str] = None,
    scene_prompt: Optional[str] = None,
) -> CharacterAnnotation:
    """
    Run Gemini 2.5 Pro on a single frame and return a CharacterAnnotation.

    Parameters
    ----------
    image_path:
        Path to the frame image.
    shot_id:
        Optional identifier (e.g., "Scene 1 / Shot 3"). If provided, it is
        injected into the prompt and the resulting annotation.
    scene_prompt:
        Optional high-level text description of the scene (from the script or
        FIBO JSON). This helps Gemini focus on the right character.

    Raises
    ------
    FileNotFoundError
        If image_path does not exist.
    RuntimeError
        If Gemini output cannot be parsed or mapped into CharacterAnnotation.
    """
    path = _as_path(image_path)
    img = _load_image(path)

    system_prompt = _build_prompt(scene_prompt=scene_prompt, shot_id=shot_id)
    model = _get_model()

    logger.debug(
        "Calling Gemini %s for character annotation (shot_id=%s, image=%s)",
        MODEL_NAME,
        shot_id,
        path,
    )

    resp = model.generate_content(
        [system_prompt, img],
        generation_config={
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 32,
            "response_mime_type": "application/json",
        },
        safety_settings=None,  # rely on Bria / platform safety layers
    )

    raw_data = _parse_gemini_json(resp)
    data = _ensure_defaults(_normalize_keys(raw_data))

    if shot_id is not None:
        data["shot_id"] = shot_id
    if scene_prompt is not None and "scene_prompt" not in data:
        data["scene_prompt"] = scene_prompt

    try:
        ann = CharacterAnnotation(**data)
    except ValidationError as e:
        logger.error("CharacterAnnotation validation failed: %s", e)
        raise RuntimeError(
            f"Failed to parse Gemini annotation into CharacterAnnotation: {e}\nRaw: {data}"
        )

    return ann


def annotate_batch(
    image_paths: Sequence[Union[str, Path]],
    shot_ids: Optional[Sequence[str]] = None,
    scene_prompts: Optional[Sequence[Optional[str]]] = None,
) -> List[CharacterAnnotation]:
    """
    Convenience helper: run annotation over multiple frames.

    This simply loops over `annotate_character_image` so that calling code
    does not need to manage per-shot IDs / prompts.

    Parameters
    ----------
    image_paths:
        List of frame paths.
    shot_ids:
        Optional list of shot identifiers (same length as image_paths).
        If shorter or None, missing entries are treated as None.
    scene_prompts:
        Optional list of scene prompts (same length as image_paths).
        If shorter or None, missing entries are treated as None.
    """
    results: List[CharacterAnnotation] = []

    shot_ids = list(shot_ids) if shot_ids is not None else []
    scene_prompts = list(scene_prompts) if scene_prompts is not None else []

    for idx, p in enumerate(image_paths):
        sid = shot_ids[idx] if idx < len(shot_ids) else None
        sprompt = scene_prompts[idx] if idx < len(scene_prompts) else None
        ann = annotate_character_image(p, shot_id=sid, scene_prompt=sprompt)
        results.append(ann)

    return results