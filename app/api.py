from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.creative_director import CreativeDirectorAgent
from agents.cinematography_agent import CinematographyAgent
from agents.continuity_agent import ContinuityAgent
from agents.qc_agent import QualityControlAgent
from agents.reviewer_agent import ReviewerAgent

import fibo.fibo_builder as fibo_builder
from fibo.image_generator_bria import FIBOBriaImageGenerator

# Local SVD renderer (optional; used by /render-mv)
from video.svd_renderer import render_mv_from_plan

# External video backend (fal.ai LongCat image→video) used by /render-mv-json
from video.video_backend import render_music_video

# Bria HTTP tools (Upscale + RMBG via Bria APIs; no local GPU)
from .bria_tools import (
    upscale_image_file,
    rmbg_image_file,
    BriaConfigError,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Environment / API key setup
# -------------------------------------------------------------------------

load_dotenv()

# Prefer BRIA_API_TOKEN, but fall back to BRIA_API_KEY for backwards compat
BRIA_API_KEY = os.getenv("BRIA_API_TOKEN") or os.getenv("BRIA_API_KEY")

if not BRIA_API_KEY:
    raise RuntimeError(
        "BRIA_API_TOKEN / BRIA_API_KEY is not set. "
        "Please add it to your .env file or environment."
    )

# -------------------------------------------------------------------------
# FastAPI app + agent singletons
# -------------------------------------------------------------------------

app = FastAPI(title="Autonomous Studio Director API")

creative_director = CreativeDirectorAgent()
cinematography_agent = CinematographyAgent()
continuity_agent = ContinuityAgent()
qc_agent = QualityControlAgent()
reviewer_agent = ReviewerAgent()

ROOT = Path(__file__).resolve().parents[1]

# Core generated directory + tool-specific subfolders
GENERATED_DIR = ROOT / "generated"
UPSCALED_DIR = GENERATED_DIR / "upscaled"
RMBG_DIR = GENERATED_DIR / "rmbg"

for d in (GENERATED_DIR, UPSCALED_DIR, RMBG_DIR):
    d.mkdir(exist_ok=True, parents=True)

# FIBO → Bria image generator
# All advanced controllability (HDR / 16-bit intent, camera geometry,
# lighting blueprints, film stock palettes, composition presets, pose
# blueprints, mood/material/continuity hints, etc.) is encoded inside
# the FIBO JSON (fibo_builder) and interpreted here.
fibo_image_generator = FIBOBriaImageGenerator(
    api_key=BRIA_API_KEY,
    output_dir=str(GENERATED_DIR),
)

# -------------------------------------------------------------------------
# Helper: unified FIBO JSON builder (supports new sh_to_fibo_json name)
# -------------------------------------------------------------------------


def shot_to_fibo_struct(shot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a FIBO StructuredPrompt JSON for a single shot.

    We support both:
      - fibo_builder.shot_to_fibo_json(shot)
      - fibo_builder.sh_to_fibo_json(shot)

    so the rest of the app (and Streamlit UI) does not care which
    function name the builder exposes.

    The returned FIBO JSON is expected to already include advanced
    controllability fields (with defaults if not explicitly set), e.g.:

      - hdr_16bit / hdr_mode
      - camera_preset / lens_focal_length / depth_of_field hints
      - lighting_preset / lighting_blueprint
      - film_palette / film_stock_palette
      - composition_preset
      - pose_blueprint
      - mood_intensity / mood_controls
      - material_style / material_controls
      - continuity_group_id
    """
    if hasattr(fibo_builder, "sh_to_fibo_json"):
        # Newer helper name (shorter) used by some versions
        return fibo_builder.sh_to_fibo_json(shot)  # type: ignore[attr-defined]

    # Backwards compatible name
    return fibo_builder.shot_to_fibo_json(shot)


# -------------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------------


class ScriptRequest(BaseModel):
    """Basic request for all text→JSON pipelines.

    For now we keep this minimal (script_text only) to stay compatible
    with the Streamlit UI. More advanced toggles (HDR, film stock, etc.)
    are injected via:
      - the shot dictionaries created/modified in the UI, and
      - fibo_builder + FIBOBriaImageGenerator.
    """

    script_text: str


class RenderMVRequest(BaseModel):
    """Used by the local SVD-based renderer (/render-mv).

    plan_path is relative to project root.
    Default: generated/mv_video_plan.json (what the Streamlit UI exports).
    """

    plan_path: str = "generated/mv_video_plan.json"


class RenderMVResponse(BaseModel):
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None


class RenderMVFromPlanRequest(BaseModel):
    """Used by the external video backend (fal.ai LongCat, etc.).

    The Streamlit UI sends the whole mv_video_plan as a dict.
    """

    plan: Dict[str, Any]


class RenderMVFromPlanResponse(BaseModel):
    status: str
    message: str
    mv_url: Optional[str] = None
    clips: Optional[List[Dict[str, Any]]] = None
    backend: Optional[str] = None
    num_clips: Optional[int] = None


class ImagePathRequest(BaseModel):
    """Request payload for Bria Upscale / RMBG tools.

    image_path: relative to repo root or absolute path.
    scale: optional scale factor (for Upscale only).
    """

    image_path: str
    scale: Optional[int] = 2


class ImageToolResponse(BaseModel):
    tool: str
    input_path: str
    output_path: str


# -------------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple health check for monitoring / liveness probes."""

    return {"status": "ok"}


# -------------------------------------------------------------------------
# Step-by-step endpoints (script → shots → cinematography)
# -------------------------------------------------------------------------


@app.post("/script-to-shots")
def script_to_shots(req: ScriptRequest) -> Dict[str, Any]:
    """Step 1 only: run CreativeDirectorAgent to get raw shot templates."""

    shots = creative_director.script_to_shots(req.script_text)
    return {"shots": shots}


@app.post("/script-to-shots-with-cinematography")
def script_to_shots_with_cinematography(req: ScriptRequest) -> Dict[str, Any]:
    """Step 2: Director + Cinematography agent to add camera + lighting."""

    base_shots: List[Dict[str, Any]] = creative_director.script_to_shots(
        req.script_text
    )
    enriched_shots = cinematography_agent.enrich_shots(base_shots)
    return {"shots": enriched_shots}


# -------------------------------------------------------------------------
# Full JSON-native pipeline (no images)
# -------------------------------------------------------------------------


@app.post("/full-pipeline")
def full_pipeline(req: ScriptRequest) -> Dict[str, Any]:
    """Full JSON-native, agentic pipeline (no images).

    Steps:
      1. CreativeDirectorAgent  -> shot templates
      2. CinematographyAgent    -> add camera + lighting
      3. ContinuityAgent        -> enforce character & scene consistency
      4. QualityControlAgent    -> generate QC report
      5. ReviewerAgent          -> wrap report into review summary
    """

    # 1) shot breakdown
    shots_step1 = creative_director.script_to_shots(req.script_text)

    # 2) camera + lighting
    shots_step2 = cinematography_agent.enrich_shots(shots_step1)

    # 3) continuity (characters + scene-level environment)
    shots_step3 = continuity_agent.apply_continuity(shots_step2)

    # 4) quality control
    qc_report = qc_agent.check_shots(shots_step3)

    # 5) reviewer summary
    review = reviewer_agent.review(shots_step3, qc_report)

    return {
        "shots": shots_step3,
        "qc_report": qc_report,
        "review": review,
        "character_bible": continuity_agent.get_character_bible(),
        "scene_environment_bible": continuity_agent.get_scene_environment_bible(),
    }


# -------------------------------------------------------------------------
# Full pipeline + FIBO JSON (no images)
# -------------------------------------------------------------------------


@app.post("/full-pipeline-fibo-json")
def full_pipeline_fibo_json(req: ScriptRequest) -> Dict[str, Any]:
    """Full agent pipeline + FIBO JSON builder (no image rendering).

    This is useful for debugging and for any external consumer that wants
    the full Bria FIBO structured prompts for each shot.
    """

    # 1) shot breakdown
    shots_step1 = creative_director.script_to_shots(req.script_text)

    # 2) camera + lighting
    shots_step2 = cinematography_agent.enrich_shots(shots_step1)

    # 3) continuity (characters + scene-level environment)
    shots_step3 = continuity_agent.apply_continuity(shots_step2)

    # 4) quality control
    qc_report = qc_agent.check_shots(shots_step3)

    # 5) reviewer summary
    review = reviewer_agent.review(shots_step3, qc_report)

    # 6) build FIBO JSON payloads-per-shot
    fibo_payloads: List[Dict[str, Any]] = []
    for shot in shots_step3:
        fibo_payloads.append(
            {
                "shot_id": shot.get("shot_id", "UNKNOWN"),
                "scene_id": shot.get("scene", 1),
                "fibo_json": shot_to_fibo_struct(shot),
            }
        )

    return {
        "shots": shots_step3,
        "qc_report": qc_report,
        "review": review,
        "character_bible": continuity_agent.get_character_bible(),
        "scene_environment_bible": continuity_agent.get_scene_environment_bible(),
        "fibo_payloads": fibo_payloads,
    }


# -------------------------------------------------------------------------
# Full pipeline + FIBO JSON + REAL Bria image generation
# -------------------------------------------------------------------------


@app.post("/full-pipeline-generate-images")
def full_pipeline_generate_images(req: ScriptRequest) -> Dict[str, Any]:
    """Full agent pipeline + FIBO JSON builder + Bria image generation.

    - Runs all agents
    - Builds FIBO JSON for each shot
    - Calls FIBOBriaImageGenerator to create PNGs per shot in `generated/`

    Advanced controllability (HDR, camera geometry, lighting blueprints,
    film stock palettes, composition / pose presets, etc.) is encoded
    inside the FIBO JSON by fibo_builder / shot_to_fibo_struct and then
    interpreted inside FIBOBriaImageGenerator. This endpoint just
    orchestrates.
    """

    # 1) shot breakdown
    shots_step1 = creative_director.script_to_shots(req.script_text)

    # 2) camera + lighting
    shots_step2 = cinematography_agent.enrich_shots(shots_step1)

    # 3) continuity (characters + scene-level environment)
    shots_step3 = continuity_agent.apply_continuity(shots_step2)

    # 4) quality control
    qc_report = qc_agent.check_shots(shots_step3)

    # 5) reviewer summary
    review = reviewer_agent.review(shots_step3, qc_report)

    # 6) build FIBO JSON payloads + generate images via Bria
    fibo_payloads: List[Dict[str, Any]] = []
    image_results: List[Dict[str, Any]] = []

    for shot in shots_step3:
        fibo_json = shot_to_fibo_struct(shot)

        fibo_payloads.append(
            {
                "shot_id": shot.get("shot_id", "UNKNOWN"),
                "scene_id": shot.get("scene", 1),
                "fibo_json": fibo_json,
            }
        )

        try:
            # FIBOBriaImageGenerator internally reads HDR / controllability
            # hints from fibo_json (hdr_mode, camera_preset,
            # lighting_preset, film_palette, composition_preset,
            # pose_blueprint, etc.).
            img_path = fibo_image_generator.generate_image_from_fibo_json(fibo_json)
        except Exception as exc:  # pragma: no cover - external service failure
            # Bubble up as 502 because it's an upstream service error (Bria API)
            raise HTTPException(
                status_code=502,
                detail=(
                    "Error generating image for shot "
                    f"{shot.get('shot_id', 'UNKNOWN')}: {exc}"
                ),
            ) from exc

        image_results.append(
            {
                "shot_id": shot.get("shot_id", "UNKNOWN"),
                "scene_id": shot.get("scene", 1),
                "image_path": img_path,
            }
        )

    return {
        "shots": shots_step3,
        "qc_report": qc_report,
        "review": review,
        "character_bible": continuity_agent.get_character_bible(),
        "scene_environment_bible": continuity_agent.get_scene_environment_bible(),
        "fibo_payloads": fibo_payloads,
        "generated_images": image_results,
    }


# -------------------------------------------------------------------------
# Local SVD music video rendering from mv_video_plan.json (optional)
# -------------------------------------------------------------------------


@app.post("/render-mv", response_model=RenderMVResponse)
def render_mv_endpoint(
    req: RenderMVRequest,
    background_tasks: BackgroundTasks,
) -> RenderMVResponse:
    """Trigger music video rendering from mv_video_plan.json using SVD.

    This will:
      - Load the plan JSON (script→shots→images→mv_video_plan.json)
      - Render per-shot clips with Stable Video Diffusion
      - Concatenate them into generated/final_music_video.mp4

    The heavy work runs in a background task so the request returns quickly.
    """

    plan_path = ROOT / req.plan_path

    if not plan_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video plan file not found at {plan_path}",
        )

    def _do_render() -> None:
        try:
            logger.info("Starting MV rendering job (local SVD)...")
            result = render_mv_from_plan(plan_path, GENERATED_DIR)
            logger.info("MV rendering completed: %s", result)
        except Exception as e:  # pragma: no cover - logging only
            logger.exception("Error in MV rendering background task: %s", e)

    background_tasks.add_task(_do_render)

    return RenderMVResponse(
        status="started",
        message=(
            "Music video rendering started on backend (local SVD). "
            "Refresh the Streamlit UI after some time to see clips and the final MV."
        ),
        result=None,
    )


# -------------------------------------------------------------------------
# External video backend rendering from in-memory mv_video_plan (fal.ai)
# -------------------------------------------------------------------------


@app.post("/render-mv-json", response_model=RenderMVFromPlanResponse)
def render_mv_from_plan_endpoint(
    req: RenderMVFromPlanRequest,
) -> RenderMVFromPlanResponse:
    """Delegate an mv_video_plan dict to the external video backend.

    Currently this uses video_backend.render_music_video(...), which is
    implemented via fal.ai's LongCat image-to-video endpoint. The backend
    is expected to return:

      - status
      - message
      - mv_url: final stitched MV URL (if applicable)
      - clips: per-shot clip metadata (including video.url, etc.)
    """

    try:
        result = render_music_video(req.plan)
    except Exception as e:  # pragma: no cover - defensive
        # Wrap any unexpected error from the video backend
        raise HTTPException(status_code=500, detail=f"Video backend error: {e}")

    # Optional: derive mv_url from the first clip if not explicitly provided
    mv_url = result.get("mv_url")
    clips = result.get("clips") or []
    if not mv_url and clips:
        first = clips[0] or {}
        mv_url = (
            first.get("video_url")
            or first.get("url")
            or (first.get("video") or {}).get("url")
        )

    return RenderMVFromPlanResponse(
        status=result.get("status", "done"),
        message=result.get("message", "Music video rendered."),
        mv_url=mv_url,
        clips=clips,
        backend=result.get("backend"),
        num_clips=result.get("num_clips"),
    )


# -------------------------------------------------------------------------
# Bria HTTP tools: Upscale + RMBG (no local GPU)
# -------------------------------------------------------------------------


@app.post("/tools/bria/upscale", response_model=ImageToolResponse)
def bria_upscale(req: ImagePathRequest) -> ImageToolResponse:
    """
    Upscale an existing image on disk using Bria's Upscale API.

    Request:
        {
          "image_path": "generated/storyboard/shot_001.png",
          "scale": 2
        }

    Response:
        {
          "tool": "upscale",
          "input_path": "...",
          "output_path": "generated/upscaled/shot_001_x2.png"
        }
    """
    # Resolve to absolute path (allow both relative and absolute)
    if os.path.isabs(req.image_path):
        src_path = Path(req.image_path)
    else:
        src_path = (ROOT / req.image_path).resolve()

    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {src_path}")

    scale = req.scale or 2

    try:
        out_bytes = upscale_image_file(src_path, scale=scale)
    except BriaConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Bria Upscale API failed: {e}",
        )

    out_name = f"{src_path.stem}_x{scale}.png"
    out_path = UPSCALED_DIR / out_name
    out_path.write_bytes(out_bytes)

    try:
        rel_in = str(src_path.relative_to(ROOT))
    except ValueError:
        rel_in = str(src_path)

    try:
        rel_out = str(out_path.relative_to(ROOT))
    except ValueError:
        rel_out = str(out_path)

    return ImageToolResponse(
        tool="upscale",
        input_path=rel_in,
        output_path=rel_out,
    )


@app.post("/tools/bria/rmbg", response_model=ImageToolResponse)
def bria_rmbg(req: ImagePathRequest) -> ImageToolResponse:
    """
    Remove background from an existing image on disk via Bria RMBG API.

    The `scale` field is ignored for now but kept in the schema for symmetry.
    """
    if os.path.isabs(req.image_path):
        src_path = Path(req.image_path)
    else:
        src_path = (ROOT / req.image_path).resolve()

    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {src_path}")

    try:
        out_bytes = rmbg_image_file(src_path)
    except BriaConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Bria RMBG API failed: {e}",
        )

    out_name = f"{src_path.stem}_rmbg.png"
    out_path = RMBG_DIR / out_name
    out_path.write_bytes(out_bytes)

    try:
        rel_in = str(src_path.relative_to(ROOT))
    except ValueError:
        rel_in = str(src_path)

    try:
        rel_out = str(out_path.relative_to(ROOT))
    except ValueError:
        rel_out = str(out_path)

    return ImageToolResponse(
        tool="rmbg",
        input_path=rel_in,
        output_path=rel_out,
    )