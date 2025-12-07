# video/svd_renderer.py

"""
Stable Video Diffusion renderer for music video clips.

This module is intended to run on a GPU-enabled server alongside your FastAPI app.
It reads an mv_video_plan.json plus generated storyboard frames and produces:

- generated/mv_clips/scene{scene}_{shot_id}.mp4  (per-shot clips)
- generated/final_music_video.mp4                 (concatenated MV)

Usage (from FastAPI):

    from pathlib import Path
    from video.svd_renderer import render_mv_from_plan

    ROOT = Path(__file__).resolve().parents[1]
    GENERATED_DIR = ROOT / "generated"

    render_mv_from_plan(GENERATED_DIR / "mv_video_plan.json", GENERATED_DIR)
"""

from __future__ import annotations

import json
import logging
import subprocess
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default model ID for Stable Video Diffusion img2vid.
# Can be overridden at runtime using the SVD_MODEL_ID environment variable.
DEFAULT_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid"

# Global pipeline cache so we only load once
_PIPELINE: StableVideoDiffusionPipeline | None = None


# ---------------------------------------------------------------------
# Motion style → SVD parameters
# ---------------------------------------------------------------------
def motion_style_to_svd_params(motion_style: str, duration_sec: float) -> Dict[str, Any]:
    """
    Map motion_style + duration to Stable Video Diffusion parameters.

    This is similar to your Colab code, but tuned a bit for a GPU server.

    - We assume SVD img2vid can comfortably handle up to ~14 frames.
    - target_fps ~ 8 for smooth but not too long clips.
    """
    target_fps = 8.0
    # SVD img2vid: 14 frames max. Clamp by duration and model constraints.
    desired_frames = int(max(8, min(14, duration_sec * target_fps)))

    style = (motion_style or "").strip()

    if style == "slow_cinematic_push_in":
        motion_bucket_id = 96
        noise_aug_strength = 0.05
    elif style == "slow_dolly_out":
        motion_bucket_id = 112
        noise_aug_strength = 0.06
    elif style == "handheld_drift":
        motion_bucket_id = 160
        noise_aug_strength = 0.12
    elif style == "orbital_around_subject":
        motion_bucket_id = 192
        noise_aug_strength = 0.16
    elif style == "static_camera_subtle_motion":
        motion_bucket_id = 64
        noise_aug_strength = 0.03
    else:
        motion_bucket_id = 128
        noise_aug_strength = 0.08

    return {
        "num_frames": desired_frames,
        "fps": int(target_fps),
        "motion_bucket_id": motion_bucket_id,
        "noise_aug_strength": noise_aug_strength,
    }


# ---------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------
def get_svd_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
) -> StableVideoDiffusionPipeline:
    """
    Lazily load and cache the Stable Video Diffusion pipeline.

    device:
        - "cuda" for GPU (recommended)
        - "cpu" is possible but very slow
        If None, auto-detect GPU if available.
    """
    global _PIPELINE

    if _PIPELINE is not None:
        return _PIPELINE

    if device is None:
        device = os.getenv("SVD_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

    effective_model_id = os.getenv("SVD_MODEL_ID", model_id)
    effective_device = device
    torch_dtype = torch.float16 if effective_device == "cuda" else torch.float32

    logger.info(
        f"Loading Stable Video Diffusion pipeline: {effective_model_id}, device={effective_device}"
    )
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        effective_model_id,
        torch_dtype=torch_dtype,
    )
    pipe.to(effective_device)

    # Optional small memory optimizations if available
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")

    _PIPELINE = pipe
    logger.info("Stable Video Diffusion pipeline loaded and cached.")
    return _PIPELINE


# ---------------------------------------------------------------------
# Clip rendering
# ---------------------------------------------------------------------
def _resolve_image_path(image_path: str, generated_dir: Path) -> Path:
    """
    Resolve image_path which may be absolute or relative to the project root.

    Assumes that generated_dir is ROOT / "generated".
    """
    img = Path(image_path)
    if img.is_absolute() and img.exists():
        return img

    # Treat as relative to project root
    project_root = generated_dir.parent
    candidate = project_root / img
    if candidate.exists():
        return candidate

    # Fall back to generated_dir itself (just in case)
    candidate2 = generated_dir / img.name
    if candidate2.exists():
        return candidate2

    # If nothing matched, return original path and let caller raise
    return img


def render_clip_for_shot(
    shot: Dict[str, Any],
    generated_dir: Path,
    pipe: StableVideoDiffusionPipeline | None = None,
    decode_chunk_size: int = 8,
) -> Path:
    """
    Render a single clip for a given shot entry from the mv_video_plan.

    Expects shot dict fields:
        - scene (int)
        - shot_id (str)
        - image_path (str)
        - duration_sec (float)
        - motion_style (str)
    """
    if pipe is None:
        pipe = get_svd_pipeline()

    scene = shot.get("scene", 1)
    shot_id = shot.get("shot_id", "SHOT")
    duration_sec = float(shot.get("duration_sec", 4.0))
    motion_style = shot.get("motion_style", "slow_cinematic_push_in")
    description = shot.get("description", "")

    if "image_path" not in shot:
        raise ValueError(f"Shot {shot_id} is missing 'image_path'")

    image_path = _resolve_image_path(str(shot["image_path"]), generated_dir)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image for shot {shot_id} not found at {image_path}"
        )

    clips_dir = generated_dir / "mv_clips"
    clips_dir.mkdir(exist_ok=True, parents=True)
    out_path = clips_dir / f"scene{scene}_{shot_id}.mp4"

    logger.info(
        f"Rendering clip for scene={scene}, shot_id={shot_id}, "
        f"desc={description[:80]!r}, duration={duration_sec}, "
        f"motion_style={motion_style}"
    )

    svd_params = motion_style_to_svd_params(motion_style, duration_sec)
    num_frames = svd_params["num_frames"]
    fps = svd_params["fps"]
    motion_bucket_id = svd_params["motion_bucket_id"]
    noise_aug_strength = svd_params["noise_aug_strength"]

    init_image = load_image(str(image_path))

    device = pipe.device
    logger.info(
        f"SVD params: frames={num_frames}, fps={fps}, "
        f"bucket={motion_bucket_id}, noise={noise_aug_strength}, device={device}"
    )

    with torch.no_grad():
        result = pipe(
            init_image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
        )
        frames = result.frames[0]

    export_to_video(frames, str(out_path), fps=fps)
    logger.info(f"Saved clip for scene={scene}, shot={shot_id} -> {out_path}")
    return out_path


# ---------------------------------------------------------------------
# Full MV rendering (clips + concat)
# ---------------------------------------------------------------------
def _concat_clips_ffmpeg(clips: List[Path], output_path: Path) -> None:
    """
    Concatenate MP4 clips into a single video using ffmpeg.

    Requires ffmpeg to be installed in the server environment.
    """
    if not clips:
        raise ValueError("No clips to concatenate")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    concat_list = output_path.parent / "concat_list.txt"

    with concat_list.open("w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output_path),
    ]
    logger.info(f"Running ffmpeg concat: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"Final MV concatenated to {output_path}")


def render_mv_from_plan(plan_path: Path, generated_dir: Path) -> Dict[str, Any]:
    """
    Render all clips and a final music video from an mv_video_plan.json file.

    plan_path:
        Path to mv_video_plan.json (as exported from Streamlit).
    generated_dir:
        Path to the 'generated' directory where images and videos should live.

    Returns:
        A dict containing:
        {
            "plan_path": str(plan_path),
            "clips": [str(...), ...],
            "final_mv_path": str(...),
            "total_shots": int,
        }
    """
    if not plan_path.exists():
        raise FileNotFoundError(f"Video plan file not found: {plan_path}")

    logger.info(f"Loading video plan from {plan_path}")
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    shots: List[Dict[str, Any]] = plan.get("shots", [])
    logger.info(f"Plan has {len(shots)} shot(s)")

    pipe = get_svd_pipeline()
    clips: List[Path] = []

    for shot in shots:
        try:
            clip_path = render_clip_for_shot(shot, generated_dir, pipe=pipe)
            clips.append(clip_path)
        except Exception as e:
            logger.exception(f"Error rendering clip for shot {shot.get('shot_id')}: {e}")

    if not clips:
        raise RuntimeError("No clips were successfully rendered")

    final_mv_path = generated_dir / "final_music_video.mp4"
    _concat_clips_ffmpeg(clips, final_mv_path)

    return {
        "plan_path": str(plan_path),
        "clips": [str(c) for c in clips],
        "final_mv_path": str(final_mv_path),
        "total_shots": len(shots),
        "success": True,
    }