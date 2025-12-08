import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import fal_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------------------------------
# Constants / Config
# -------------------------------------------------

# LongCat Image → Video model on fal.ai
LONGCAT_MODEL_ID = "fal-ai/longcat-video/image-to-video/480p"

# Hard cap for how long each shot should be (seconds).
# We use this to avoid super long & expensive clips.
MAX_DURATION_SEC_PER_SHOT: float = 2.0

# Default FPS for LongCat; used to derive num_frames when we override duration.
DEFAULT_FPS: int = 15

# Max parallel LongCat jobs at once
MAX_PARALLEL_JOBS: int = 3


# -------------------------------------------------
# Env + helpers
# -------------------------------------------------
def _ensure_fal_key() -> None:
    """
    Ensure fal.ai API key is present.

    fal_client reads FAL_KEY by default, but we also support FAL_API_KEY
    as an alias for convenience in local .env files.
    """
    fal_key = os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY")
    if not fal_key:
        raise RuntimeError(
            "Missing FAL_KEY or FAL_API_KEY in environment.\n"
            "Set it in your `.env`, e.g.\n"
            "  FAL_API_KEY=sk-XXXXXXXXXXXXXXXX\n"
            "or\n"
            "  FAL_KEY=sk-XXXXXXXXXXXXXXXX"
        )

    # Mirror FAL_API_KEY into FAL_KEY if only that is set
    if not os.getenv("FAL_KEY"):
        os.environ["FAL_KEY"] = fal_key


def _duration_to_frames(duration_sec: float, fps: int = DEFAULT_FPS) -> int:
    """
    Convert duration (seconds) to approximate frame count for LongCat.

    We cap at a sensible minimum so very short durations still produce
    enough motion to feel like a real shot.
    """
    try:
        sec = float(duration_sec)
    except Exception:
        sec = MAX_DURATION_SEC_PER_SHOT

    frames = int(round(sec * fps))
    return max(8, frames)


def _clamp_duration(raw_duration: Optional[float]) -> float:
    """
    Take whatever duration is in the plan and clamp it to:
        0.5s <= duration <= MAX_DURATION_SEC_PER_SHOT

    This guarantees that we don't accidentally ask LongCat to generate
    huge 10–15s clips for each shot.
    """
    if raw_duration is None:
        sec = MAX_DURATION_SEC_PER_SHOT
    else:
        try:
            sec = float(raw_duration)
        except Exception:
            sec = MAX_DURATION_SEC_PER_SHOT

    # Clamp to [0.5, MAX_DURATION_SEC_PER_SHOT]
    if sec < 0.5:
        sec = 0.5
    if sec > MAX_DURATION_SEC_PER_SHOT:
        sec = MAX_DURATION_SEC_PER_SHOT

    return sec


# -------------------------------------------------
# Internal: single-shot LongCat call
# -------------------------------------------------
def _render_single_shot_with_longcat(
    shot: Dict[str, Any],
    idx: int,
    total: int,
) -> Optional[Dict[str, Any]]:
    """
    Upload one frame & call LongCat for a single shot.

    Returns:
        clip dict on success, or None on failure.
    """
    scene_id = shot.get("scene")
    shot_id = shot.get("shot_id", f"shot_{idx}")
    description = (shot.get("description") or "").strip()
    image_path = shot.get("image_path")

    if not image_path:
        logger.warning("[LongCat] Shot %s has no image_path, skipping", shot_id)
        return None

    local_path = Path(image_path)
    if not local_path.exists():
        logger.warning(
            "[LongCat] Image path does not exist on disk for shot %s: %s",
            shot_id,
            local_path,
        )
        return None

    # Clamp duration and derive frames.
    raw_duration = shot.get("duration_sec")
    effective_duration = _clamp_duration(raw_duration)
    fps = DEFAULT_FPS
    num_frames = _duration_to_frames(effective_duration, fps=fps)

    logger.info(
        "[LongCat][%d/%d] Scene %s, shot %s: uploading frame '%s' "
        "(requested=%.2fs, effective=%.2fs → %d frames @ %dfps)",
        idx,
        total,
        scene_id,
        shot_id,
        local_path,
        float(raw_duration) if raw_duration is not None else MAX_DURATION_SEC_PER_SHOT,
        effective_duration,
        num_frames,
        fps,
    )

    # 1) Upload local image file to fal storage -> get public URL
    try:
        image_url = fal_client.upload_file(str(local_path))
    except Exception as e:
        logger.exception("[LongCat] Failed to upload file for shot %s: %s", shot_id, e)
        return None

    logger.info(
        "[LongCat][%d/%d] Scene %s, shot %s: uploaded to %s, calling LongCat "
        "(frames=%d, fps=%d)",
        idx,
        total,
        scene_id,
        shot_id,
        image_url,
        num_frames,
        fps,
    )

    # Fallback prompt if description is empty
    fal_prompt = (
        description
        or "cinematic music video shot, slow motion, moody lighting, high quality"
    )

    # 2) Call LongCat image→video
    def _on_queue_update(update: Any) -> None:
        from fal_client import InProgress  # lazy import

        if isinstance(update, InProgress):
            for log in update.logs:
                msg = log.get("message")
                if msg:
                    logger.info("[fal][%s] %s", shot_id, msg)

    try:
        result = fal_client.subscribe(
            LONGCAT_MODEL_ID,
            arguments={
                "image_url": image_url,
                "prompt": fal_prompt,
                # 🔹 Force ~2s per shot: num_frames & fps
                "num_frames": num_frames,
                "fps": fps,
                # Safety + quality knobs (using reasonable defaults)
                "enable_safety_checker": True,
                "video_output_type": "X264 (.mp4)",
                "video_quality": "high",
                "video_write_mode": "balanced",
            },
            with_logs=True,
            on_queue_update=_on_queue_update,
        )
    except Exception as e:
        logger.exception(
            "[LongCat] Generation failed for shot %s (scene %s): %s",
            shot_id,
            scene_id,
            e,
        )
        return None

    # 3) Extract video URL from result
    data: Any = getattr(result, "data", result)
    video_info: Any = None

    if isinstance(data, dict):
        # Expected shape: {"video": {"url": "...", ...}, "prompt": "...", ...}
        video_info = data.get("video") or data.get("output") or data
    else:
        logger.warning(
            "[LongCat] Unexpected result type for shot %s: %r",
            shot_id,
            type(data),
        )

    video_url: Optional[str] = None
    if isinstance(video_info, dict):
        video_url = video_info.get("url") or video_info.get("video_url")

    if not video_url:
        logger.warning(
            "[LongCat] No video URL returned for shot %s. Raw data: %r",
            shot_id,
            data,
        )
        return None

    logger.info(
        "[LongCat][%d/%d] Scene %s, shot %s: received video URL: %s",
        idx,
        total,
        scene_id,
        shot_id,
        video_url,
    )

    return {
        "scene": scene_id,
        "shot_id": shot_id,
        "video_url": video_url,
        "prompt": fal_prompt,
        "image_url": image_url,
        "duration_sec": effective_duration,
        "fps": fps,
        "num_frames": num_frames,
    }


# -------------------------------------------------
# Core: render_music_video (parallel LongCat)
# -------------------------------------------------
def render_music_video(
    plan: Dict[str, Any],
    progress_cb: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> Dict[str, Any]:
    """
    High-level entrypoint used by FastAPI (/render-mv-json).

    Currently, this ALWAYS uses fal.ai LongCat Image→Video:
        model id: 'fal-ai/longcat-video/image-to-video/480p'

    For each storyboard shot, we:
      1. Upload the generated frame (PNG) to fal storage.
      2. Call LongCat image→video with the shot description as prompt.
      3. Collect the returned video URL.

    ✅ New:
      * Calls are executed in PARALLEL using ThreadPoolExecutor,
        with MAX_PARALLEL_JOBS workers.
      * There is an optional progress callback:
            progress_cb(completed, total, shot_id)
        which you can use in FastAPI or Streamlit to drive a progress bar.

    Expected `plan` format (built in Streamlit `build_video_plan()`):

        {
            "project_type": "music_video",
            "video_backend": "...",   # currently ignored, we always use LongCat
            "created_at": "...",
            "total_shots": N,
            "shots": [
                {
                    "scene": 1,
                    "shot_id": "S1_SHOT1",
                    "description": "...",
                    "duration_sec": 4.0,
                    "motion_style": "slow_cinematic_push_in",
                    "image_path": "generated/fibo_xxx.png",
                },
                ...
            ]
        }

    This function **does not concatenate** clips into a single MP4.
    Instead, it returns one LongCat video per shot:

        {
            "backend": "fal-ai/longcat-video/image-to-video/480p",
            "mv_url": "<url-of-first-clip-or-empty-string>",
            "clips": [
                {
                    "scene": 1,
                    "shot_id": "S1_SHOT1",
                    "video_url": "...",
                    "prompt": "...",
                    "image_url": "https://fal-storage/....png",
                    "duration_sec": 2.0,
                    "fps": 15,
                    "num_frames": 30
                },
                ...
            ],
            "num_clips": <int>
        }

    Notes:
      * Every shot is clamped to about **2 seconds** to save credits.
        (see MAX_DURATION_SEC_PER_SHOT).
      * We explicitly pass `num_frames` and `fps` into LongCat so we
        don't get the long (~11s) default clips.
      * LongCat calls are parallelized (up to MAX_PARALLEL_JOBS).
      * If all shots fail, `clips` will be [] and `mv_url` == "".
    """
    _ensure_fal_key()

    shots: List[Dict[str, Any]] = plan.get("shots", [])
    if not shots:
        logger.warning("render_music_video: plan has no shots")
        return {
            "backend": LONGCAT_MODEL_ID,
            "mv_url": "",
            "clips": [],
            "num_clips": 0,
        }

    total = len(shots)
    clips: List[Dict[str, Any]] = []

    logger.info(
        "Starting LongCat image-to-video rendering for %d shots using model %s (max_parallel=%d)",
        total,
        LONGCAT_MODEL_ID,
        MAX_PARALLEL_JOBS,
    )

    # ThreadPoolExecutor for parallel LongCat jobs
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        future_to_shot_id: Dict[Any, str] = {}
        for idx, shot in enumerate(shots, start=1):
            shot_id = shot.get("shot_id", f"shot_{idx}")
            future = executor.submit(_render_single_shot_with_longcat, shot, idx, total)
            future_to_shot_id[future] = shot_id

        completed = 0
        for future in as_completed(future_to_shot_id):
            shot_id = future_to_shot_id[future]
            clip = None
            try:
                clip = future.result()
            except Exception as e:
                logger.exception(
                    "Unhandled exception while rendering shot %s: %s", shot_id, e
                )

            if clip is not None:
                clips.append(clip)
            else:
                logger.warning("Shot %s produced no clip.", shot_id)

            completed += 1
            progress_fraction = completed / total
            logger.info(
                "LongCat rendering progress: %d/%d (%.1f%%)",
                completed,
                total,
                progress_fraction * 100.0,
            )

            # Optional callback for real-time progress (for Streamlit / FastAPI)
            if progress_cb is not None:
                try:
                    progress_cb(completed, total, shot_id)
                except Exception as cb_err:
                    logger.warning("progress_cb raised an exception: %s", cb_err)

    # For now, choose the first successful clip as "mv_url" representative
    mv_url = clips[0]["video_url"] if clips else ""

    result_payload: Dict[str, Any] = {
        "backend": LONGCAT_MODEL_ID,
        "mv_url": mv_url,
        "clips": clips,
        "num_clips": len(clips),
        # Optional: echo some plan metadata for debugging / UI
        "project_type": plan.get("project_type"),
        "created_at": plan.get("created_at"),
    }

    logger.info(
        "render_music_video complete: %d clip(s) generated, mv_url=%s",
        len(clips),
        mv_url or "<empty>",
    )

    return result_payload