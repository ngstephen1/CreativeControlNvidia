from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import logging
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from agents.creative_director import CreativeDirectorAgent
from agents.cinematography_agent import CinematographyAgent
from agents.continuity_agent import ContinuityAgent
from agents.qc_agent import QualityControlAgent
from agents.reviewer_agent import ReviewerAgent
import fibo.fibo_builder as fibo_builder  # module with shot_to_fibo_json
from fibo.image_generator_bria import FIBOBriaImageGenerator
from video.svd_renderer import render_mv_from_plan
from video.video_backend import render_music_video_with_pika
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

# fibo_builder is the imported module (fibo.fibo_builder) with shot_to_fibo_json
fibo_image_generator = FIBOBriaImageGenerator(
    api_key=BRIA_API_KEY,
    output_dir=str(GENERATED_DIR),
)

class ScriptRequest(BaseModel):
    script_text: str


class RenderMVRequest(BaseModel):
    # Path to mv_video_plan.json relative to project root.
    # Default matches your Streamlit export location.
    plan_path: str = "generated/mv_video_plan.json"


class RenderMVResponse(BaseModel):
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None

class RenderMVFromPlanRequest(BaseModel):
    plan: Dict[str, Any]

class RenderMVFromPlanResponse(BaseModel):
    status: str
    message: str
    mv_url: Optional[str] = None
    clips: Optional[List[Dict[str, Any]]] = None
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
    """
    Step 1 only: run CreativeDirectorAgent to get raw shot templates.
    """
    shots = creative_director.script_to_shots(req.script_text)
    return {"shots": shots}


@app.post("/script-to-shots-with-cinematography")
def script_to_shots_with_cinematography(req: ScriptRequest) -> Dict[str, Any]:
    """
    Step 2: run CreativeDirectorAgent, then CinematographyAgent
    to enrich each shot with camera + lighting JSON.
    """
    base_shots: List[Dict[str, Any]] = creative_director.script_to_shots(req.script_text)
    enriched_shots = cinematography_agent.enrich_shots(base_shots)
    return {"shots": enriched_shots}


# -------------------------------------------------------------------------
# Full JSON-native pipeline (no images)
# -------------------------------------------------------------------------


@app.post("/full-pipeline")
def full_pipeline(req: ScriptRequest) -> Dict[str, Any]:
    """
    Full JSON-native, agentic pipeline (no images):

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

    # 5) reviewer summary (later: AI suggestions)
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
    """
    Full agent pipeline + FIBO JSON builder.

    1. CreativeDirectorAgent  -> shot templates
    2. CinematographyAgent    -> add camera + lighting
    3. ContinuityAgent        -> enforce character & scene consistency
    4. QualityControlAgent    -> generate QC report
    5. ReviewerAgent          -> wrap report into review summary
    6. fibo.fibo_builder      -> map each shot to FIBO-style JSON payload
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
                "fibo_json": fibo_builder.shot_to_fibo_json(shot),
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
    """
    Full agent pipeline + FIBO JSON builder + Bria FIBO image generation.

    - runs all agents
    - builds FIBO JSON for each shot
    - calls FIBOBriaImageGenerator to create PNGs per shot in `generated/`
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
        fibo_json = fibo_builder.shot_to_fibo_json(shot)
        fibo_payloads.append(
            {
                "shot_id": shot.get("shot_id", "UNKNOWN"),
                "scene_id": shot.get("scene", 1),
                "fibo_json": fibo_json,
            }
        )

        try:
            img_path = fibo_image_generator.generate_image_from_fibo_json(fibo_json)
        except Exception as exc:  # pragma: no cover - broad to surface API errors
            raise HTTPException(
                status_code=502,
                detail=f"Error generating image for shot "
                f"{shot.get('shot_id', 'UNKNOWN')}: {exc}",
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
    
    
    
@app.post("/render-mv", response_model=RenderMVResponse)
def render_mv_endpoint(
    req: RenderMVRequest,
    background_tasks: BackgroundTasks,
) -> RenderMVResponse:
    """
    Trigger music video rendering from mv_video_plan.json.

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

    def _do_render():
        try:
            logger.info("Starting MV rendering job...")
            result = render_mv_from_plan(plan_path, GENERATED_DIR)
            logger.info(f"MV rendering completed: {result}")
        except Exception as e:
            logger.exception(f"Error in MV rendering background task: {e}")

    background_tasks.add_task(_do_render)

    return RenderMVResponse(
        status="started",
        message=(
            "Music video rendering started on backend. "
            "Refresh the Streamlit UI after some time to see clips and the final MV."
        ),
        result=None,
    )
    
@app.post("/render-mv-json", response_model=RenderMVFromPlanResponse)
def render_mv_from_plan_endpoint(req: RenderMVFromPlanRequest) -> RenderMVFromPlanResponse:
    """
    Accept an mv_video_plan dict and delegate to the external video backend
    (Pika, Runway, fal.ai, etc.) via video_backend.render_music_video(...).

    The backend is expected to return a final MV URL and optionally per-shot
    clip URLs.
    """
    try:
        result = render_music_video(req.plan)
    except Exception as e:
        # Wrap any unexpected error from the video backend
        raise HTTPException(status_code=500, detail=f"Video backend error: {e}")

    return RenderMVFromPlanResponse(
        status=result.get("status", "done"),
        message=result.get("message", "Music video rendered."),
        mv_url=result.get("mv_url"),
        clips=result.get("clips"),
    )