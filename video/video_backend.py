import os
import logging
from pathlib import Path
from typing import Any, Dict, List

import fal_client

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# LongCat Image → Video model on fal.ai
LONGCAT_MODEL_ID = "fal-ai/longcat-video/image-to-video/480p"


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


def _duration_to_frames(duration_sec: float, fps: int = 15) -> int:
    """
    Convert duration (seconds) to approximate frame count for LongCat.

    We cap at a sensible minimum so very short durations still produce
    enough motion to feel like a real shot.
    """
    try:
        sec = float(duration_sec)
    except Exception:
        sec = 3.0  # safe default

    frames = int(round(sec * fps))
    return max(8, frames)


# -------------------------------------------------
# Core: render_music_video
# -------------------------------------------------
def render_music_video(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-level entrypoint used by FastAPI (/render-mv-json).

    Currently, this ALWAYS uses fal.ai LongCat Image→Video:
        model id: 'fal-ai/longcat-video/image-to-video/480p'

    For each storyboard shot, we:
      1. Upload the generated frame (PNG) to fal storage.
      2. Call LongCat image→video with the shot description as prompt.
      3. Collect the returned video URL.

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

    Returns a JSON-serializable dict:

        {
            "backend": "fal-ai/longcat-video/image-to-video/480p",
            "mv_url": "<url-of-first-clip-or-empty-string>",
            "clips": [
                {
                    "scene": 1,
                    "shot_id": "S1_SHOT1",
                    "video_url": "...",
                    "prompt": "...",
                    "image_url": "https://fal-storage/....png"
                },
                ...
            ],
            "num_clips": <int>
        }

    Notes:
      * We always return an `mv_url` field, even if it's an empty string.
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

    clips: List[Dict[str, Any]] = []

    logger.info(
        "Starting LongCat image-to-video rendering for %d shots using model %s",
        len(shots),
        LONGCAT_MODEL_ID,
    )

    for idx, shot in enumerate(shots, start=1):
        scene_id = shot.get("scene")
        shot_id = shot.get("shot_id", f"shot_{idx}")
        description = shot.get("description", "").strip()
        image_path = shot.get("image_path")

        if not image_path:
            logger.warning("Shot %s has no image_path, skipping", shot_id)
            continue

        local_path = Path(image_path)
        if not local_path.exists():
            logger.warning(
                "Image path does not exist on disk for shot %s: %s",
                shot_id,
                local_path,
            )
            continue

        # Derive approximate frames from duration
        duration_sec = float(shot.get("duration_sec", 4.0))
        fps = 15
        num_frames = _duration_to_frames(duration_sec, fps=fps)

        logger.info(
            "Scene %s, shot %s: uploading frame '%s' (duration=%.2fs → %d frames)",
            scene_id,
            shot_id,
            local_path,
            duration_sec,
            num_frames,
        )

        # 1) Upload local image file to fal storage -> get public URL
        try:
            image_url = fal_client.upload_file(str(local_path))
        except Exception as e:
            logger.exception("Failed to upload file for shot %s: %s", shot_id, e)
            continue

        logger.info(
            "Scene %s, shot %s: uploaded to %s, calling LongCat (frames=%d, fps=%d)",
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
            # Optional: stream logs while the job is running
            from fal_client import InProgress

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
                    # Optional: you can also set num_frames, fps explicitly.
                    # The LongCat endpoint has its own defaults; we keep it simple
                    # and lean on those defaults for now.
                    # "num_frames": num_frames,
                    # "fps": fps,
                    "enable_safety_checker": True,
                    "video_output_type": "X264 (.mp4)",
                    "video_quality": "high",
                    "video_write_mode": "balanced",
                },
                with_logs=True,
                on_queue_update=_on_queue_update,
            )
        except Exception as e:
            logger.exception("LongCat generation failed for shot %s: %s", shot_id, e)
            continue

        # 3) Extract video URL from result
        # According to fal docs, `result` is usually a dict with a "video" field.
        data: Any = getattr(result, "data", result)
        video_info: Any = None

        if isinstance(data, dict):
            # Expected shape: {"video": {"url": "...", ...}, "prompt": "...", ...}
            video_info = data.get("video") or data.get("output") or data
        else:
            logger.warning(
                "Unexpected LongCat result type for shot %s: %r", shot_id, type(data)
            )

        video_url: str | None = None
        if isinstance(video_info, dict):
            video_url = video_info.get("url") or video_info.get("video_url")

        if not video_url:
            logger.warning(
                "No video URL returned for shot %s. Raw data: %r", shot_id, data
            )
            continue

        logger.info(
            "Scene %s, shot %s: received video URL from LongCat: %s",
            scene_id,
            shot_id,
            video_url,
        )

        clips.append(
            {
                "scene": scene_id,
                "shot_id": shot_id,
                "video_url": video_url,
                "prompt": fal_prompt,
                "image_url": image_url,
            }
        )

    mv_url = clips[0]["video_url"] if clips else ""

    result_payload: Dict[str, Any] = {
        "backend": LONGCAT_MODEL_ID,
        "mv_url": mv_url,
        "clips": clips,
        "num_clips": len(clips),
    }

    logger.info(
        "render_music_video complete: %d clip(s) generated, mv_url=%s",
        len(clips),
        mv_url or "<empty>",
    )

    return result_payload