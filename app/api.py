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

import fibo.fibo_builder as fibo_builder  # module with shot_to_fibo_json / sh_to_fibo_json
from fibo.image_generator_bria import FIBOBriaImageGenerator

# Local SVD renderer (optional; used by /render-mv)
from video.svd_renderer import render_mv_from_plan

# External video backend (fal.ai LongCat image→video)
from video.video_backend import render_music_video


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
GENERATED_DIR = ROOT / "generated"
GENERATED_DIR.mkdir(exist_ok=True, parents=True)

# FIBO → Bria image generator (HDR/controllability is handled in the generator
# and in fibo_builder via the FIBO JSON; api.py stays simple)
fibo_image_generator = FIBOBriaImageGenerator(
    api_key=BRIA_API_KEY,
    output_dir=str(GENERATED_DIR),
)


# -------------------------------------------------------------------------
# Helper: unified FIBO JSON builder (supports new sh_to_fibo_json name)
# -------------------------------------------------------------------------

def shot_to_fibo_struct(shot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a FIBO StructuredPrompt JSON for a single shot.

    We support both the original `shot_to_fibo_json(shot)` API and a newer
    `sh_to_fibo_json(shot)` variant, so that the Streamlit UI and other
    consumers can call into `app.api` without worrying about naming.

    All advanced controllability (HDR / 16-bit pipeline, camera geometry,
    lighting blueprints, film stock palettes, etc.) is encoded inside the
    returned FIBO JSON by fibo_builder.
    """
    if hasattr(fibo_builder, "sh_to_fibo_json"):
        return fibo_builder.sh_to_fibo_json(shot)  # type: ignore[attr-defined]
    return fibo_builder.shot_to_fibo_json(shot)


# -------------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------------


class ScriptRequest(BaseModel):
    """Basic request for all text→JSON pipelines.

    For now we keep this minimal (script_text only) to stay compatible with
    the Streamlit UI. More advanced toggles (HDR, film stock, etc.) are
    injected via the FIBO JSON and Bria generator, not through this schema.
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


# -------------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# -------------------------------------------------------------------------
# Step-by-step endpoints
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
    """Full JSON-native, agentic pipeline (no images):

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
    """Full agent pipeline + FIBO JSON builder + Bria FIBO image generation.

    - runs all agents
    - builds FIBO JSON for each shot
    - calls FIBOBriaImageGenerator to create PNGs per shot in `generated/`

    Advanced controllability (HDR, camera geometry, lighting blueprints,
    film stock palettes, etc.) is encoded inside the FIBO JSON by
    fibo_builder / shot_to_fibo_struct and interpreted inside
    FIBOBriaImageGenerator. This endpoint just orchestrates.
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
            # hints from the fibo_json (e.g. hdr_mode, lighting_blueprint,
            # film_stock_palette, etc.), so we don't need extra args here.
            img_path = fibo_image_generator.generate_image_from_fibo_json(fibo_json)
        except Exception as exc:
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
    """Trigger music video rendering from mv_video_plan.json using the local
    Stable Video Diffusion renderer (video/svd_renderer.py).

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
    """Accept an mv_video_plan dict and delegate to the external video backend
    via video_backend.render_music_video(...).

    The backend (currently fal.ai LongCat image→video) is expected to return:
      - status
      - message
      - mv_url: final stitched MV URL
      - clips: per-shot clip metadata (including clip_url / video.url, etc.)
    """

    try:
        result = render_music_video(req.plan)
    except Exception as e:
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