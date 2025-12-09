from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import logging
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from agents.creative_director import CreativeDirectorAgent
from agents.cinematography_agent import CinematographyAgent
from agents.continuity_agent import ContinuityAgent
from agents.qc_agent import QualityControlAgent
from agents.reviewer_agent import ReviewerAgent

import fibo.fibo_builder as fibo_builder

# Gemini character continuity annotator (vision + JSON)
try:
    from gemini.character_annotator import (
        annotate_character_image,
        annotate_batch,
        CharacterAnnotation,
    )

    HAS_GEMINI = True
except Exception:  # pragma: no cover - optional dependency
    HAS_GEMINI = False
    annotate_character_image = None  # type: ignore[assignment]
    annotate_batch = None  # type: ignore[assignment]
    CharacterAnnotation = None  # type: ignore[assignment]

from fibo.image_generator_bria import FIBOBriaImageGenerator
from PIL import Image, ImageFilter, ImageEnhance

# Local SVD renderer (optional; used by /render-mv)
try:
    from video.svd_renderer import render_mv_from_plan  # type: ignore

    HAS_LOCAL_SVD = True
except Exception:
    # On Streamlit Cloud (or minimal installs) torch/diffusers may not be present.
    # We still want the app to boot; /render-mv will just be disabled.
    HAS_LOCAL_SVD = False
    render_mv_from_plan = None  # type: ignore[assignment]

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

# If set to 1, true, or True, this forces the /tools/bria/upscale endpoint
# to use the local PIL-based stub instead of Bria's cloud API.
BRIA_USE_LOCAL_UPSCALE = os.getenv("BRIA_USE_LOCAL_UPSCALE", "0").strip() in {
    "1",
    "true",
    "True",
}

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
# Helper: unified FIBO JSON builder (supports sh_to_fibo_json alias)
# -------------------------------------------------------------------------


def shot_to_fibo_struct(shot: Dict[str, Any]) -> Dict[str, Any]:
    """Return a FIBO StructuredPrompt JSON for a single shot.

    We support both:
      - fibo_builder.shot_to_fibo_json(shot)
      - fibo_builder.sh_to_fibo_json(shot)

    so the rest of the app (and Streamlit UI) does not care which
    function name the builder exposes.
    """

    if hasattr(fibo_builder, "sh_to_fibo_json"):
        # Newer helper name (shorter) used by some versions
        return fibo_builder.sh_to_fibo_json(shot)  # type: ignore[attr-defined]

    # Backwards compatible name
    return fibo_builder.shot_to_fibo_json(shot)


# -------------------------------------------------------------------------
# Small helper to handle Pydantic v1/v2 dict conversion
# -------------------------------------------------------------------------


def _model_to_dict(model: Any) -> Dict[str, Any]:
    """Convert a Pydantic model to a plain dict for both v1 and v2."""

    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    # Fallback – should not normally happen
    return dict(model)  # type: ignore[arg-type]


# -------------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------------


class ScriptRequest(BaseModel):
    """Basic request for all text→JSON pipelines."""

    script_text: str


class RenderMVRequest(BaseModel):
    """Used by the local SVD-based renderer (/render-mv)."""

    # Path relative to project root
    plan_path: str = "generated/mv_video_plan.json"


class RenderMVResponse(BaseModel):
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None


class RenderMVFromPlanRequest(BaseModel):
    """Used by the external video backend (fal.ai LongCat, etc.)."""

    plan: Dict[str, Any]


class RenderMVFromPlanResponse(BaseModel):
    status: str
    message: str
    mv_url: Optional[str] = None
    clips: Optional[List[Dict[str, Any]]] = None
    backend: Optional[str] = None
    num_clips: Optional[int] = None


class ImagePathRequest(BaseModel):
    """Request payload for Bria Upscale / RMBG tools."""

    image_path: str
    scale: Optional[int] = 2


class ImageToolResponse(BaseModel):
    tool: str
    input_path: str
    output_path: str


# -------------------------------------------------------------------------
# Gemini character annotation models
# -------------------------------------------------------------------------


class CharacterAnnotateRequest(BaseModel):
    """Request for per-shot Gemini character annotation."""

    image_path: str
    shot_id: Optional[str] = None
    scene_prompt: Optional[str] = None


class CharacterAnnotateBatchRequest(BaseModel):
    """Batch version of character annotation."""

    items: List[CharacterAnnotateRequest]


class CharacterAnnotateResponse(BaseModel):
    """Standard response wrapper around CharacterAnnotation."""

    status: str
    annotation: Dict[str, Any]


class CharacterContinuityRequest(BaseModel):
    """High-level request used by the Continuity Inspector UI."""

    image_paths: List[str]
    shot_ids: Optional[List[Optional[str]]] = None
    scene_prompts: Optional[List[Optional[str]]] = None


class CharacterContinuityResponse(BaseModel):
    """Response payload for the Continuity Inspector."""

    status: str
    annotations: List[Dict[str, Any]]


# -------------------------------------------------------------------------
# Local "Bria Upscaler" stub (PIL-based resize, separate endpoint)
# -------------------------------------------------------------------------


class BriaUpscaleRequest(BaseModel):
    image_path: str
    scale: int = 2  # 2x, 4x, etc.


class BriaUpscaleResponse(BaseModel):
    status: str
    message: str
    input_path: str
    output_path: str
    scale: int
    width: int
    height: int


def _local_upscale(image_path: str, scale: int) -> Path:
    """Enhanced CPU-only upscaler using PIL, with fake AI enhancement."""

    if scale <= 1:
        raise ValueError("scale must be > 1")

    src = Path(image_path)
    if not src.is_absolute():
        src = ROOT / src
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input image not found: {src}")

    dst = GENERATED_DIR / "upscaled" / f"upscaled_x{scale}_{src.name}"
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        new_size = (w * scale, h * scale)
        im = im.resize(new_size, Image.LANCZOS)
        im = im.filter(ImageFilter.DETAIL)
        im = im.filter(ImageFilter.SHARPEN)
        im = ImageEnhance.Contrast(im).enhance(1.15)
        im = ImageEnhance.Color(im).enhance(1.1)
        im.save(dst)

    return dst


@app.post("/tools/bria/upscale-local", response_model=BriaUpscaleResponse)
def bria_upscale_local(req: BriaUpscaleRequest) -> BriaUpscaleResponse:
    """Temporary local 'upscale' that doesn't call Bria's cloud API."""

    try:
        out_path = _local_upscale(req.image_path, req.scale)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("Error in local upscaler: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Local upscaler failed: {e}",
        )

    with Image.open(out_path) as im:
        w, h = im.size

    return BriaUpscaleResponse(
        status="ok",
        message=(
            "Locally upscaled image (PIL stub). Replace with real Bria API when ready."
        ),
        input_path=req.image_path,
        output_path=str(out_path.relative_to(ROOT)),
        scale=req.scale,
        width=w,
        height=h,
    )


# -------------------------------------------------------------------------
# ComfyUI export models
# -------------------------------------------------------------------------


class ComfyUIExportRequest(BaseModel):
    """Request to build a ComfyUI graph template from storyboard shots."""

    shots: List[Dict[str, Any]]
    fibo_payloads: Optional[List[Dict[str, Any]]] = None


class ComfyUIExportResponse(BaseModel):
    """Response wrapper around the ComfyUI graph JSON."""

    status: str
    message: str
    graph: Dict[str, Any]
    num_shots: int
    template_type: Optional[str] = None
    version: Optional[str] = None


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
    """Full JSON-native, agentic pipeline (no images)."""

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
    """Full agent pipeline + FIBO JSON builder (no image rendering)."""

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
    """Full agent pipeline + FIBO JSON builder + Bria image generation."""

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
            img_path = fibo_image_generator.generate_image_from_fibo_json(fibo_json)
        except Exception as exc:  # pragma: no cover - external service failure
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
    """Trigger music video rendering from mv_video_plan.json using local SVD."""

    if not HAS_LOCAL_SVD:
        raise HTTPException(
            status_code=501,
            detail=(
                "Local SVD renderer is not available on this deployment "
                "(missing torch/diffusers). Use /render-mv-json instead."
            ),
        )

    plan_path = ROOT / req.plan_path

    if not plan_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video plan file not found at {plan_path}",
        )

    def _do_render() -> None:
        try:
            logger.info("Starting MV rendering job (local SVD)...")
            result = render_mv_from_plan(plan_path, GENERATED_DIR)  # type: ignore[arg-type]
            logger.info("MV rendering completed: %s", result)
        except Exception as e:  # pragma: no cover - logging only
            logger.exception("Error in MV rendering background task: %s", e)

    background_tasks.add_task(_do_render)

    return RenderMVResponse(
        status="started",
        message=(
            "Music video rendering started on backend (local SVD). "
            "On this machine it may take several minutes."
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
    """Delegate an mv_video_plan dict to the external video backend."""

    try:
        result = render_music_video(req.plan)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Video backend error: {e}")

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
# ComfyUI export endpoint
# -------------------------------------------------------------------------


@app.post("/tools/bria/export-comfyui", response_model=ComfyUIExportResponse)
def export_comfyui(req: ComfyUIExportRequest) -> ComfyUIExportResponse:
    """Build a ComfyUI graph template from storyboard shots + FIBO JSON."""

    shots = req.shots or []
    if not shots:
        raise HTTPException(status_code=400, detail="shots list is empty")

    if req.fibo_payloads:
        fibo_payloads = req.fibo_payloads
    else:
        fibo_payloads = []
        for shot in shots:
            fibo_json = shot_to_fibo_struct(shot)
            fibo_payloads.append(
                {
                    "shot_id": shot.get("shot_id", "UNKNOWN"),
                    "scene_id": shot.get("scene", 1),
                    "fibo_json": fibo_json,
                }
            )

    try:
        graph = fibo_builder.build_comfyui_template(shots, fibo_payloads)
    except Exception as e:
        logger.exception("Error building ComfyUI template: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build ComfyUI template: {e}",
        )

    out_dir = GENERATED_DIR / "comfyui"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "comfyui_graph.json"
    try:
        out_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    except Exception as e:  # non-fatal
        logger.exception("Failed to write ComfyUI graph to disk: %s", e)

    template_type = graph.get("template_type") if isinstance(graph, dict) else None
    version = graph.get("version") if isinstance(graph, dict) else None

    return ComfyUIExportResponse(
        status="ok",
        message="ComfyUI graph template built successfully.",
        graph=graph,
        num_shots=len(shots),
        template_type=template_type,
        version=version,
    )


# -------------------------------------------------------------------------
# Bria HTTP tools: Upscale + RMBG (no local GPU)
# -------------------------------------------------------------------------


@app.post("/tools/bria/upscale", response_model=ImageToolResponse)
def bria_upscale(req: ImagePathRequest) -> ImageToolResponse:
    """Upscale an existing image on disk using Bria's Upscale API."""

    if os.path.isabs(req.image_path):
        src_path = Path(req.image_path)
    else:
        src_path = (ROOT / req.image_path).resolve()

    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {src_path}")

    scale = req.scale or 2

    if BRIA_USE_LOCAL_UPSCALE:
        try:
            out_path = _local_upscale(str(src_path), scale)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Local upscaler failed: {e}")

        try:
            rel_in = str(src_path.relative_to(ROOT))
        except ValueError:
            rel_in = str(src_path)
        try:
            rel_out = str(out_path.relative_to(ROOT))
        except ValueError:
            rel_out = str(out_path)

        return ImageToolResponse(
            tool="upscale_local",
            input_path=rel_in,
            output_path=rel_out,
        )

    try:
        out_bytes = upscale_image_file(src_path, scale=scale)
    except Exception as e:
        logger.warning("Bria upscale failed, falling back to local: %s", e)
        try:
            out_path = _local_upscale(str(src_path), scale)
        except Exception as le:
            raise HTTPException(
                status_code=500, detail=f"Local upscaler failed: {le}"
            )

        try:
            rel_in = str(src_path.relative_to(ROOT))
        except ValueError:
            rel_in = str(src_path)
        try:
            rel_out = str(out_path.relative_to(ROOT))
        except ValueError:
            rel_out = str(out_path)

        return ImageToolResponse(
            tool="upscale_local",
            input_path=rel_in,
            output_path=rel_out,
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
    """Remove background from an existing image via Bria RMBG API."""

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


# -------------------------------------------------------------------------
# Gemini tools: Character continuity annotation
# -------------------------------------------------------------------------


@app.post("/tools/gemini/annotate-character", response_model=CharacterAnnotateResponse)
def gemini_annotate_character(
    req: CharacterAnnotateRequest,
) -> CharacterAnnotateResponse:
    """Run Gemini 2.5 Pro over a single frame to annotate the main character."""

    if not HAS_GEMINI:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini annotator is not available on this deployment. "
                "Ensure GEMINI_API_KEY is set and gemini/character_annotator.py "
                "is importable."
            ),
        )

    if os.path.isabs(req.image_path):
        src_path = Path(req.image_path)
    else:
        src_path = (ROOT / req.image_path).resolve()

    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {src_path}")

    try:
        ann_model = annotate_character_image(
            image_path=str(src_path),
            shot_id=req.shot_id,
            scene_prompt=req.scene_prompt,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        logging.exception("Gemini annotation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Gemini annotation failed: {e}")

    annotation_dict = _model_to_dict(ann_model)
    return CharacterAnnotateResponse(status="ok", annotation=annotation_dict)


@app.post(
    "/tools/gemini/annotate-character-batch",
    response_model=List[CharacterAnnotateResponse],
)
def gemini_annotate_character_batch(
    req: CharacterAnnotateBatchRequest,
) -> List[CharacterAnnotateResponse]:
    """Batch Gemini character annotation for multiple frames."""

    if not HAS_GEMINI:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini annotator is not available on this deployment. "
                "Ensure GEMINI_API_KEY is set and gemini/character_annotator.py "
                "is importable."
            ),
        )

    if not req.items:
        raise HTTPException(status_code=400, detail="items list is empty")

    paths: List[str] = []
    shot_ids: List[Optional[str]] = []
    prompts: List[Optional[str]] = []
    for item in req.items:
        if os.path.isabs(item.image_path):
            p = Path(item.image_path)
        else:
            p = (ROOT / item.image_path).resolve()
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {p}")
        paths.append(str(p))
        shot_ids.append(item.shot_id)
        prompts.append(item.scene_prompt)

    try:
        ann_models = annotate_batch(paths, shot_ids=shot_ids, scene_prompts=prompts)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        logging.exception("Gemini batch annotation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Gemini batch annotation failed: {e}",
        )

    responses: List[CharacterAnnotateResponse] = []
    for ann in ann_models:
        annotation_dict = _model_to_dict(ann)
        responses.append(CharacterAnnotateResponse(status="ok", annotation=annotation_dict))

    return responses


# -------------------------------------------------------------------------
# Gemini continuity endpoint for Streamlit UI
# -------------------------------------------------------------------------


@app.post(
    "/tools/gemini/character-continuity",
    response_model=CharacterContinuityResponse,
)
def gemini_character_continuity(
    req: CharacterContinuityRequest,
) -> CharacterContinuityResponse:
    """High-level Gemini 2.5 Pro continuity endpoint used by the Streamlit UI."""

    if not HAS_GEMINI:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini annotator is not available on this deployment. "
                "Ensure GEMINI_API_KEY is set and gemini/character_annotator.py "
                "is importable."
            ),
        )

    if not req.image_paths:
        raise HTTPException(status_code=400, detail="image_paths list is empty")

    num_items = len(req.image_paths)
    shot_ids_list = (req.shot_ids or []) + [None] * max(
        0, num_items - len(req.shot_ids or [])
    )
    scene_prompts_list = (req.scene_prompts or []) + [None] * max(
        0, num_items - len(req.scene_prompts or [])
    )

    resolved_paths: List[str] = []
    used_shot_ids: List[Optional[str]] = []
    used_prompts: List[Optional[str]] = []

    for idx in range(num_items):
        raw_path = req.image_paths[idx]
        if os.path.isabs(raw_path):
            p = Path(raw_path)
        else:
            p = (ROOT / raw_path).resolve()
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {p}")
        resolved_paths.append(str(p))
        used_shot_ids.append(shot_ids_list[idx])
        used_prompts.append(scene_prompts_list[idx])

    try:
        ann_models = annotate_batch(
            resolved_paths,
            shot_ids=used_shot_ids,
            scene_prompts=used_prompts,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive
        logging.exception("Gemini continuity endpoint failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Gemini continuity endpoint failed: {e}",
        )

    annotations: List[Dict[str, Any]] = []
    for ann in ann_models:
        annotations.append(_model_to_dict(ann))

    return CharacterContinuityResponse(status="ok", annotations=annotations)