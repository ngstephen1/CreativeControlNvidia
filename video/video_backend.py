import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import moviepy
import requests
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

# Directory where we store downloaded clips and combined MV files
MV_OUTPUT_DIR = Path(os.getenv("MV_OUTPUT_DIR", "generated/mv_outputs"))
MV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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

    # Safe “requested” string for logging (no float() crashes)
    if isinstance(raw_duration, (int, float)):
        requested_str = f"{float(raw_duration):.2f}"
    elif raw_duration is None:
        requested_str = f"{MAX_DURATION_SEC_PER_SHOT:.2f}"
    else:
        requested_str = str(raw_duration)

    logger.info(
        "[LongCat][%d/%d] Scene %s, shot %s: uploading frame '%s' "
        "(requested=%ss, effective=%.2fs → %d frames @ %dfps)",
        idx,
        total,
        scene_id,
        shot_id,
        local_path,
        requested_str,
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
        "sequence_index": idx,
    }


def _download_clip_to_disk(video_url: str, index: int) -> Optional[Path]:
    """Download a remote video URL to MV_OUTPUT_DIR and return the local path.

    If anything fails, logs a warning and returns None.
    """
    try:
        parsed = urlparse(video_url)
        # Try to keep a meaningful file name extension when present
        ext = ".mp4"
        if parsed.path:
            _, _, maybe_name = parsed.path.rpartition("/")
            if "." in maybe_name:
                ext = "." + maybe_name.split(".")[-1]
        local_name = f"clip_{index:03d}{ext}"
        local_path = MV_OUTPUT_DIR / local_name

        resp = requests.get(video_url, stream=True, timeout=60)
        resp.raise_for_status()
        with local_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return local_path
    except Exception as e:  # pragma: no cover - best-effort utility
        logger.warning("Failed to download clip %s: %s", video_url, e)
        return None


def _concatenate_clips_locally(clips: List[Dict[str, Any]]) -> Optional[Path]:
    """Concatenate all clip video URLs into a single local MP4.

    This uses moviepy on CPU. If moviepy is not installed or
    concatenation fails, returns None and logs a warning.
    """
    if not clips:
        return None

    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except Exception as e:  # pragma: no cover - optional dependency
        logger.warning("moviepy not available, skipping concatenation: %s", e)
        return None

    # Ensure deterministic order based on sequence_index
    ordered = sorted(clips, key=lambda c: c.get("sequence_index", 9999))

    local_paths: List[Path] = []
    for i, clip in enumerate(ordered):
        url = clip.get("video_url")
        if not url:
            continue
        path = _download_clip_to_disk(url, i + 1)
        if path is not None:
            local_paths.append(path)

    if not local_paths:
        logger.warning("No local video files were downloaded; cannot concatenate.")
        return None

    video_clips = []
    try:
        for p in local_paths:
            video_clips.append(VideoFileClip(str(p)))
        final = concatenate_videoclips(video_clips, method="compose")
        output_path = MV_OUTPUT_DIR / "combined_music_video.mp4"
        # Use a sane codec; adjust if needed for your environment
        final.write_videofile(str(output_path), codec="libx264", audio_codec="aac")
        for vc in video_clips:
            vc.close()
        return output_path
    except Exception as e:  # pragma: no cover - best-effort utility
        logger.warning("Failed to concatenate clips: %s", e)
        return None
    finally:
        for vc in video_clips:
            try:
                vc.close()
            except Exception:
                pass


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
      4. Attempt to concatenate all clips into a single local MP4.

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

    Returns:

        {
            "backend": "fal-ai/longcat-video/image-to-video/480p",
            "mv_url": "<combined-local-mp4-or-first-clip-url>",
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
            "num_clips": <int>,
            "project_type": "...",
            "created_at": "..."
        }

    Notes:
      * Every shot is clamped to about **2 seconds** to save credits
        (see MAX_DURATION_SEC_PER_SHOT).
      * We explicitly pass `num_frames` and `fps` into LongCat so we
        don't get the long (~11s) default clips.
      * LongCat calls are parallelized (up to MAX_PARALLEL_JOBS).
      * We attempt to concatenate all clips into one MP4 using moviepy;
        if that fails, we fall back to the first clip URL.
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
            "project_type": plan.get("project_type"),
            "created_at": plan.get("created_at"),
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

    # Sort clips by original sequence index for a stable timeline
    clips = sorted(clips, key=lambda c: c.get("sequence_index", 9999))

    # Try to concatenate all clips into a single local MP4
    combined_path = _concatenate_clips_locally(clips)
    if combined_path is not None:
        mv_url = combined_path.as_posix()
        logger.info(
            "Combined music video created at %s (from %d clip[s])",
            mv_url,
            len(clips),
        )
    else:
        # Fallback: use the first individual clip URL if available
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