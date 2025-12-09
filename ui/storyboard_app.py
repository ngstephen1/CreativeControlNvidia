import os
import sys
import copy
import base64
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw  # for composition overlays and Gemini bounding boxes

# -------------------------------------------------
# Bootstrap: paths + env
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env so Streamlit sees BRIA_API_TOKEN etc.
load_dotenv(dotenv_path=ROOT / ".env", override=False)

BRIA_API_TOKEN = (
    os.getenv("BRIA_API_TOKEN")
    or os.getenv("BRIA_API_KEY")  # fallback if you named it differently
    or ""
)
BRIA_IMAGE_GENERATION_BASE = os.getenv(
    "BRIA_IMAGE_GENERATION_BASE", "https://engine.prod.bria-api.com/v2/image"
)
BRIA_IMAGE_EDIT_BASE = os.getenv(
    "BRIA_IMAGE_EDIT_BASE", "https://engine.prod.bria-api.com/v2/image/edit"
)

# Backend for rendering music videos (FastAPI + external video backend)
RENDER_BACKEND_BASE = os.getenv("RENDER_BACKEND_BASE", "http://localhost:8000")

GENERATED_DIR = ROOT / "generated"
ASSETS_DIR = GENERATED_DIR / "assets"
GENERATED_DIR.mkdir(exist_ok=True, parents=True)
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

# Video clips and final MV paths
VIDEO_CLIPS_DIR = GENERATED_DIR / "mv_clips"
FULL_MV_PATH = GENERATED_DIR / "final_music_video.mp4"
VIDEO_CLIPS_DIR.mkdir(exist_ok=True, parents=True)

# Import your pipeline pieces directly from FastAPI app
from app.api import (  # type: ignore[import]
    creative_director,
    cinematography_agent,
    continuity_agent,
    qc_agent,
    reviewer_agent,
    fibo_builder,
    fibo_image_generator,
)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def _find_clip_for_shot(scene_id: int, shot_id: str) -> Optional[Path]:
    """Look for an MP4 clip rendered for this scene/shot.

    Expected filename pattern: scene{scene_id}_{shot_id}.mp4 inside VIDEO_CLIPS_DIR.
    Returns the Path if it exists, otherwise None.
    """
    safe_shot_id = str(shot_id)
    candidate = VIDEO_CLIPS_DIR / f"scene{scene_id}_{safe_shot_id}.mp4"
    if candidate.exists():
        return candidate
    return None


def _ensure_bria_token() -> Optional[str]:
    """Return token or show a Streamlit error and return None."""
    if not BRIA_API_TOKEN:
        st.error(
            "BRIA_API_TOKEN is not set.\n\n"
            "Add it to your `.env` in the project root, e.g.:\n\n"
            "`BRIA_API_TOKEN=sk_XXXXXXXXXXXXXXXX`"
        )
        return None
    return BRIA_API_TOKEN


def _image_file_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def _download_image_to_path(url: str, out_dir: Path, prefix: str) -> str:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    ext = ".png"
    filename = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
    out_path = out_dir / filename
    with open(out_path, "wb") as f:
        f.write(resp.content)
    return str(out_path)


def _bria_edit_call(endpoint: str, payload: Dict[str, Any]) -> str:
    """Call a Bria v2 image-edit endpoint and return local path to downloaded image.

    `endpoint` is like 'remove_background', 'enhance', 'replace_background'.
    """
    token = _ensure_bria_token()
    if token is None:
        raise RuntimeError("BRIA_API_TOKEN missing")

    url = f"{BRIA_IMAGE_EDIT_BASE}/{endpoint}"
    headers = {"Content-Type": "application/json", "api_token": token}

    # Force sync for a simpler demo
    payload = dict(payload)
    payload.setdefault("sync", True)

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code not in (200, 202):
        raise RuntimeError(
            f"Bria edit error {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()
    # v2 edit returns {"result": {"image_url": "..."}}
    result = data.get("result", {})
    image_url = result.get("image_url")
    if not image_url:
        raise RuntimeError(f"Bria edit missing image_url: {data}")

    return _download_image_to_path(image_url, ASSETS_DIR, endpoint)


def _create_composition_grid(image_path: str, shot_id: str) -> str:
    """Create a rule-of-thirds composition overlay for the given image.

    Saves a new PNG with grid lines into ASSETS_DIR and returns its path.
    """
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        raise RuntimeError(f"Failed to open image for composition grid: {e}")

    w, h = img.size
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)

    # Rule-of-thirds positions
    v1 = w / 3
    v2 = 2 * w / 3
    h1 = h / 3
    h2 = 2 * h / 3

    # Semi-transparent white lines
    line_color = (255, 255, 255, 180)
    line_width = max(1, int(min(w, h) * 0.0025))

    # Vertical lines
    draw.line([(v1, 0), (v1, h)], fill=line_color, width=line_width)
    draw.line([(v2, 0), (v2, h)], fill=line_color, width=line_width)
    # Horizontal lines
    draw.line([(0, h1), (w, h1)], fill=line_color, width=line_width)
    draw.line([(0, h2), (w, h2)], fill=line_color, width=line_width)

    out_name = f"shot_{shot_id}_composition_grid.png"
    out_path = ASSETS_DIR / out_name
    overlay.save(out_path, format="PNG")
    return str(out_path)


def build_directors_commentary(result: Dict[str, Any]) -> str:
    """Generate a human-readable 'director's commentary' style summary
    of the storyboard, using existing structured data.
    """
    shots: List[Dict[str, Any]] = result.get("shots", [])
    fibo_payloads: List[Dict[str, Any]] = result.get("fibo_payloads", [])

    if not shots:
        return "No shots available yet – generate a storyboard first."

    # Build lookup for FIBO JSON by shot_id
    fibo_by_shot: Dict[str, Dict[str, Any]] = {}
    for fp in fibo_payloads:
        sid = str(fp.get("shot_id", "UNKNOWN"))
        fibo_by_shot[sid] = fp.get("fibo_json", {})

    # Aggregate per scene
    scenes: Dict[int, Dict[str, Any]] = {}
    for shot in shots:
        scene_id = int(shot.get("scene", 1))
        sid = str(shot.get("shot_id", "UNKNOWN"))
        fibo = fibo_by_shot.get(sid, {})

        env = shot.get("environment", {})
        pc = fibo.get("photographic_characteristics", {})
        aesth = fibo.get("aesthetics", {})

        scenes.setdefault(
            scene_id,
            {
                "shots": [],
                "lighting": set(),
                "film_stocks": set(),
                "hdr_modes": set(),
                "camera_angles": set(),
                "moods": set(),
                "color_schemes": set(),
                "locations": set(),
            },
        )

        s = scenes[scene_id]
        s["shots"].append(shot)
        if "lighting_blueprint" in fibo:
            s["lighting"].add(fibo.get("lighting_blueprint"))
        if "film_stock" in fibo:
            s["film_stocks"].add(fibo.get("film_stock"))
        if "hdr_mode" in fibo:
            s["hdr_modes"].add(fibo.get("hdr_mode"))
        if pc.get("camera_angle"):
            s["camera_angles"].add(pc.get("camera_angle"))
        if aesth.get("mood_atmosphere"):
            s["moods"].add(aesth.get("mood_atmosphere"))
        if aesth.get("color_scheme"):
            s["color_schemes"].add(aesth.get("color_scheme"))
        if env.get("location"):
            s["locations"].add(env.get("location"))

    lines: List[str] = []
    lines.append(
        "This music video is structured as a sequence of carefully controlled scenes, "
        "with FIBO governing camera language, lighting, and color so the look stays consistent from shot to shot.\n"
    )

    for scene_id in sorted(scenes.keys()):
        s = scenes[scene_id]
        shots_list = s["shots"]
        num_shots = len(shots_list)

        lighting = ", ".join(sorted(x for x in s["lighting"] if x)) or "default studio-style lighting"
        film = ", ".join(sorted(x for x in s["film_stocks"] if x)) or "no explicit film stock (neutral digital look)"
        hdr = ", ".join(sorted(x for x in s["hdr_modes"] if x)) or "standard dynamic range"
        angles = ", ".join(sorted(x for x in s["camera_angles"] if x)) or "mostly eye-level framing"
        moods = ", ".join(sorted(x for x in s["moods"] if x)) or "cinematic but understated mood"
        colors = ", ".join(sorted(x for x in s["color_schemes"] if x)) or "balanced color palette"
        locations = ", ".join(sorted(x for x in s["locations"] if x)) or "implied city locations"

        lines.append(f"Scene {scene_id}:")
        lines.append(
            f"- We use {num_shots} shot(s) to tell this moment, leaning on {angles} to shape the emotional perspective."
        )
        lines.append(
            f"- The lighting blueprint leans toward **{lighting}**, which keeps the subject readable while shaping depth and atmosphere."
        )
        lines.append(
            f"- Color is guided by **{colors}**, and film look is steered by **{film}**, which gives a cohesive treatment across the scene."
        )
        lines.append(
            f"- HDR pipeline mode is **{hdr}**, chosen to balance highlight detail against a strong, musical contrast."
        )
        lines.append(
            f"- Overall mood here is **{moods}**, anchored in the recurring sense of place: **{locations}**.\n"
        )

    lines.append(
        "Across the full storyboard, FIBO gives us JSON-level control over lens, angle, color and lighting. "
        "That means we can regenerate individual shots without losing the overall look of the video."
    )

    return "\n".join(lines)


# -------------------------------------------------
# Multi-agent storyboard pipeline
# -------------------------------------------------
def run_full_pipeline(script_text: str) -> Dict[str, Any]:
    """Mirror of /full-pipeline-generate-images but as a plain function
    for use inside Streamlit.
    """
    # 1) shot breakdown
    shots_step1 = creative_director.script_to_shots(script_text)

    # 2) camera + lighting
    shots_step2 = cinematography_agent.enrich_shots(shots_step1)

    # 3) continuity (characters + scene-level environment)
    shots_step3 = continuity_agent.apply_continuity(shots_step2)

    # 4) quality control
    qc_report = qc_agent.check_shots(shots_step3)

    # 5) reviewer summary
    review = reviewer_agent.review(shots_step3, qc_report)

    # 6) FIBO JSON + images
    fibo_payloads: List[Dict[str, Any]] = []
    image_results: List[Dict[str, Any]] = []

    for shot in shots_step3:
        # Use new fallback for sh_to_fibo_json if available
        fibo_json = (
            fibo_builder.sh_to_fibo_json(shot)
            if hasattr(fibo_builder, "sh_to_fibo_json")
            else fibo_builder.shot_to_fibo_json(shot)
        )
        fibo_payloads.append(
            {
                "shot_id": shot.get("shot_id", "UNKNOWN"),
                "scene_id": shot.get("scene", 1),
                "fibo_json": fibo_json,
            }
        )

        img_path = fibo_image_generator.generate_image_from_fibo_json(fibo_json)
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
        "fibo_payloads": fibo_payloads,
        "generated_images": image_results,
    }


# -------------------------------------------------
# Video plan builder (for music video export)
# -------------------------------------------------
def build_video_plan(result: Dict[str, Any], video_backend: str) -> Dict[str, Any]:
    """Build a JSON-serializable plan that describes which storyboard frames
    should become which video clips. This is meant to be consumed by a
    GPU backend (FastAPI + SVD) or external service (Pika, Runway, fal.ai) for
    image→video or text→video generation.
    """
    shots: List[Dict[str, Any]] = result.get("shots", [])
    images: List[Dict[str, Any]] = result.get("generated_images", [])

    shot_entries: List[Dict[str, Any]] = []
    for shot, img_meta in zip(shots, images):
        shot_entries.append(
            {
                "scene": shot.get("scene"),
                "shot_id": shot.get("shot_id"),
                "description": shot.get("description"),
                "duration_sec": float(shot.get("duration_sec", 2.0)),
                "motion_style": shot.get(
                    "motion_style", "slow_cinematic_push_in"
                ),
                "image_path": img_meta.get("image_path"),
            }
        )

    return {
        "project_type": "music_video",
        "video_backend": video_backend,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "total_shots": len(shot_entries),
        "shots": shot_entries,
    }


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Autonomous Studio Director – FIBO Storyboard",
        layout="wide",
    )

    # Shared backend base for all HTTP calls (FastAPI host)
    if "backend_base" not in st.session_state:
        st.session_state["backend_base"] = RENDER_BACKEND_BASE

    st.title("🎬 Autonomous Studio Director")
    st.caption(
        "JSON-native FIBO storyboard generator (Bria) + controllable Music Video export"
    )

    col_left, col_right = st.columns([2, 1], gap="large")

    # ------------------------
    # Left: script input
    # ------------------------
    with col_left:
        st.subheader("Script")
        default_script = (
            "Scene 1: Night rain on the city. A young Asian male violinist in his mid-20s, with medium-length black hair tucked behind his ears, warm brown eyes, a slim calm face, wearing a dark navy coat and a charcoal grey scarf, holding a worn reddish-brown wooden violin, plays under a street lamp while cars pass in the distance.\n\n"
            "Scene 2: Closer shot of the same young Asian male violinist in his mid-20s, medium-length black hair behind his ears, warm brown eyes, slim face, navy coat, grey scarf, holding the same reddish-brown violin. His eyes are closed, raindrops fall onto the instrument, and neon reflections shimmer in puddles beneath him.\n\n"
            "Scene 3: The next morning, the young Asian male violinist (mid-20s, medium-length black hair behind his ears, warm brown eyes, slim face, navy coat, grey scarf) rides a tram through the city with his reddish-brown violin resting beside him, watching the world pass by the window in slow motion.\n\n"
            "Scene 4: Inside a quiet cafe, the young Asian male violinist (mid-20s, medium-length black hair behind his ears, warm brown eyes, slim face, navy coat, grey scarf) sits by a fogged window with his reddish-brown violin, notebook open, sketching new music ideas.\n\n"
            "Scene 5: Evening rooftop performance. The young Asian male violinist (mid-20s, medium-length black hair behind his ears, warm brown eyes, slim face, navy coat, grey scarf) plays his reddish-brown violin for a small crowd under warm string lights, with the glowing city skyline behind him.\n\n"
            "Scene 6: Montage of the young Asian male violinist (mid-20s, medium-length black hair behind his ears, warm brown eyes, slim face, navy coat, grey scarf) and his reddish-brown violin: close-ups of his hands on the strings, blurred city lights, silhouettes walking in the rain, and a gentle smile toward the camera.\n\n"
            "Scene 7: Final wide shot at blue hour. The young Asian male violinist (mid-20s, medium-length black hair behind his ears, warm brown eyes, slim face, navy coat, grey scarf) stands on a bridge holding his reddish-brown violin, music echoing as cars below form light trails."
        )
        script_text = st.text_area(
            "Paste your scene description or multi-scene script:",
            value=default_script,
            height=220,
            key="script_input_area",
        )

        generate_button = st.button("🚀 Generate Storyboard", type="primary")

    # ------------------------
    # Right: settings & status
    # ------------------------
    with col_right:
        st.subheader("Run settings")
        st.markdown(
            "- Uses your local **multi-agent pipeline** (Creative Director → Cinematography → Continuity → QC → Reviewer)\n"
            "- Calls **Bria FIBO** for JSON-native image control (structured_prompt)\n"
            "- Saves keyframes to `generated/` for downstream video backends\n"
        )

        # Video backend selection (for mv_video_plan)
        backend_label_map = {
            "Stable Video Diffusion (SVD – open-source, GPU backend)": "svd",
            "LongCat Video (fal.ai – image→video)": "longcat",
        }
        default_backend_label = (
            "Stable Video Diffusion (SVD – open-source, GPU backend)"
        )
        selected_backend_label = st.selectbox(
            "Video backend for image→video (used in export plan):",
            list(backend_label_map.keys()),
            index=list(backend_label_map.keys()).index(default_backend_label),
            key="video_backend_select",
        )
        video_backend_value = backend_label_map[selected_backend_label]
        st.session_state["video_backend"] = video_backend_value

        st.markdown("**Backend service URL**")
        st.code(RENDER_BACKEND_BASE, language="text")

        if "last_run_summary" in st.session_state:
            st.markdown("**Last storyboard run:**")
            st.json(st.session_state["last_run_summary"])

        if BRIA_API_TOKEN:
            st.success("BRIA_API_TOKEN loaded.")
        else:
            st.warning("BRIA_API_TOKEN not found – Bria-powered edit tools will fail.")

    # ========================
    #  Run storyboard pipeline
    # ========================
    if generate_button:
        if not script_text.strip():
            st.error("Please enter a script first.")
            return

        with st.spinner("Generating storyboard with Bria FIBO + multi-agent pipeline…"):
            result = run_full_pipeline(script_text)

        st.session_state["storyboard"] = result
        st.session_state["last_run_summary"] = {
            "num_shots": len(result["shots"]),
            "num_images": len(result["generated_images"]),
        }

        st.success("Storyboard generated!")

    # If we have a storyboard in session, render it
    result: Dict[str, Any] | None = st.session_state.get("storyboard")
    if not result:
        return

    st.subheader("📽️ Storyboard Frames")

    shots = result["shots"]
    images = result["generated_images"]
    fibo_payloads = result["fibo_payloads"]

    # Map shot_id -> path / fibo for Shot Asset Lab & continuity inspector & Gemini
    shot_id_to_image_path: Dict[str, str] = {}
    shot_id_to_fibo: Dict[str, Dict[str, Any]] = {}

    for shot, img_meta, fibo_payload in zip(shots, images, fibo_payloads):
        shot_id = shot.get("shot_id")
        scene_id = shot.get("scene")
        if shot_id is None:
            continue
        shot_id_str = str(shot_id)
        shot_id_to_image_path[shot_id_str] = img_meta["image_path"]
        shot_id_to_fibo[shot_id_str] = fibo_payload["fibo_json"]

        st.markdown(f"### Scene {scene_id} – Shot {shot_id}")

        cols = st.columns([2, 3])

        # Left: original image + rendered clip preview + composition overlay
        with cols[0]:
            img_path = img_meta["image_path"]
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True, caption="Original frame")
            else:
                st.warning(f"Image not found: {img_path}")

            # NEW: Composition grid overlay (rule-of-thirds)
            show_grid = st.checkbox(
                "Show composition grid overlay",
                key=f"comp_grid_toggle_{shot_id_str}",
            )
            if show_grid and os.path.exists(img_path):
                grid_key = f"comp_grid_path_{shot_id_str}"
                grid_path = st.session_state.get(grid_key)
                if not grid_path or not os.path.exists(grid_path):
                    try:
                        grid_path = _create_composition_grid(img_path, shot_id_str)
                        st.session_state[grid_key] = grid_path
                    except Exception as e:
                        st.error(f"Failed to create composition grid: {e}")
                        grid_path = None
                if grid_path and os.path.exists(grid_path):
                    st.image(
                        grid_path,
                        use_container_width=True,
                        caption="Composition (rule-of-thirds overlay)",
                    )

            # Rendered clip preview (if available)
            clip_path = _find_clip_for_shot(scene_id, shot_id_str)
            if clip_path and os.path.exists(clip_path):
                st.markdown("**Rendered clip**")
                st.video(str(clip_path))
            else:
                st.caption(
                    "No rendered clip found yet for this shot. Trigger MV rendering and refresh."
                )

            # Allow sending this shot into Shot Asset Lab
            if st.button(
                "🧪 Send to Shot Asset Lab",
                key=f"send_to_lab_{shot_id_str}",
            ):
                st.session_state["asset_lab_shot_id"] = shot_id_str
                st.session_state["asset_lab_image_path"] = img_path
                st.info(f"Shot {shot_id} sent to Shot Asset Lab (below).")

        # Right: metadata + JSON + controllability
        with cols[1]:
            st.markdown("**Description**")
            st.write(shot.get("description", ""))

            st.markdown("**Camera intent**")
            st.write(shot.get("camera_intent", ""))

            st.markdown("**Environment**")
            st.json(shot.get("environment", {}))

            # 🎵 Music Video settings per shot (duration & motion style)
            with st.expander("🎵 Music Video Settings"):
                current_duration = float(shot.get("duration_sec", 2.0))
                current_motion = shot.get(
                    "motion_style", "slow_cinematic_push_in"
                )

                new_duration = st.slider(
                    "Shot duration (seconds)",
                    min_value=1.0,
                    max_value=12.0,
                    value=current_duration,
                    step=0.5,
                    key=f"dur_{shot_id_str}",
                )

                new_motion = st.selectbox(
                    "Motion style (hint for image→video backend)",
                    [
                        "slow_cinematic_push_in",
                        "slow_dolly_out",
                        "handheld_drift",
                        "orbital_around_subject",
                        "static_camera_subtle_motion",
                    ],
                    index=[
                        "slow_cinematic_push_in",
                        "slow_dolly_out",
                        "handheld_drift",
                        "orbital_around_subject",
                        "static_camera_subtle_motion",
                    ].index(current_motion)
                    if current_motion
                    in [
                        "slow_cinematic_push_in",
                        "slow_dolly_out",
                        "handheld_drift",
                        "orbital_around_subject",
                        "static_camera_subtle_motion",
                    ]
                    else 0,
                    key=f"motion_{shot_id_str}",
                )

                # Persist back into shot dict (so export + QC see updated values)
                shot["duration_sec"] = new_duration
                shot["motion_style"] = new_motion

                st.caption(
                    "These values are embedded in `mv_video_plan.json` for GPU video backends "
                    "(SVD, LongCat, etc.) to generate consistent clips."
                )

            fibo_payload = next(
                (fp for fp in fibo_payloads if fp["shot_id"] == shot_id), None
            ) or {"fibo_json": {}}

            with st.expander("FIBO StructuredPrompt JSON (original)"):
                st.json(fibo_payload["fibo_json"])

            # 🎛 Controllability panel – structured JSON tweaks
            with st.expander("🎛 Tweak & Regenerate this shot"):
                with st.form(f"regen_form_{shot_id_str}"):
                    original_fibo = fibo_payload["fibo_json"]
                    pc = original_fibo.get("photographic_characteristics", {})
                    aesth = original_fibo.get("aesthetics", {})

                    current_angle = pc.get("camera_angle", "eye-level")
                    current_mood = aesth.get(
                        "mood_atmosphere", "cinematic, contemplative atmosphere"
                    )
                    current_colors = aesth.get(
                        "color_scheme",
                        "cool blues and purples with warm highlights from artificial lights",
                    )

                    angle_options = [
                        "eye-level",
                        "high",
                        "low",
                        "bird's-eye",
                        "worm's-eye",
                    ]
                    angle_index = (
                        angle_options.index(current_angle)
                        if current_angle in angle_options
                        else angle_options.index("eye-level")
                    )

                    # 🔑 unique keys per shot for these widgets
                    new_angle = st.selectbox(
                        "Camera angle",
                        angle_options,
                        index=angle_index,
                        key=f"regen_angle_{shot_id_str}",
                    )

                    new_mood = st.selectbox(
                        "Mood / atmosphere",
                        [
                            "cinematic, contemplative atmosphere",
                            "tense, suspenseful atmosphere",
                            "hopeful, uplifting atmosphere",
                            "melancholic, lonely atmosphere",
                            "energetic, dynamic atmosphere",
                        ],
                        index=[
                            "cinematic, contemplative atmosphere",
                            "tense, suspenseful atmosphere",
                            "hopeful, uplifting atmosphere",
                            "melancholic, lonely atmosphere",
                            "energetic, dynamic atmosphere",
                        ].index(current_mood)
                        if current_mood
                        in [
                            "cinematic, contemplative atmosphere",
                            "tense, suspenseful atmosphere",
                            "hopeful, uplifting atmosphere",
                            "melancholic, lonely atmosphere",
                            "energetic, dynamic atmosphere",
                        ]
                        else 0,
                        key=f"regen_mood_{shot_id_str}",
                    )

                    new_colors = st.selectbox(
                        "Color scheme",
                        [
                            "cool blues and purples with warm highlights from artificial lights",
                            "warm orange and teal blockbuster palette",
                            "soft neutral daylight colors",
                            "high-contrast noir with deep shadows",
                        ],
                        index=[
                            "cool blues and purples with warm highlights from artificial lights",
                            "warm orange and teal blockbuster palette",
                            "soft neutral daylight colors",
                            "high-contrast noir with deep shadows",
                        ].index(current_colors)
                        if current_colors
                        in [
                            "cool blues and purples with warm highlights from artificial lights",
                            "warm orange and teal blockbuster palette",
                            "soft neutral daylight colors",
                            "high-contrast noir with deep shadows",
                        ]
                        else 0,
                        key=f"regen_colors_{shot_id_str}",
                    )

                    # --- HDR / 16-bit toggle ---
                    hdr_mode_current = original_fibo.get("hdr_mode", "off")
                    hdr_options = ["off", "hdr10", "hdr10_plus"]
                    hdr_mode = st.selectbox(
                        "HDR / Color pipeline",
                        hdr_options,
                        index=hdr_options.index(hdr_mode_current)
                        if hdr_mode_current in hdr_options
                        else 0,
                        key=f"regen_hdr_{shot_id_str}",
                    )

                    # --- Lighting blueprint presets ---
                    lighting_presets = [
                        "default",
                        "soft_key_fill_back",
                        "high_contrast_noir",
                        "neon_night_city",
                        "golden_hour_backlight",
                    ]
                    lighting_current = original_fibo.get(
                        "lighting_blueprint", "default"
                    )
                    lighting_blueprint = st.selectbox(
                        "Lighting blueprint",
                        lighting_presets,
                        index=lighting_presets.index(lighting_current)
                        if lighting_current in lighting_presets
                        else 0,
                        key=f"regen_light_{shot_id_str}",
                    )

                    # --- Film stock color palettes ---
                    film_stocks = [
                        "none",
                        "kodak_2383_cine",
                        "fuji_pro_400h",
                        "ilford_hp5_bw",
                    ]
                    film_current = original_fibo.get("film_stock", "none")
                    film_stock = st.selectbox(
                        "Film stock / color palette",
                        film_stocks,
                        index=film_stocks.index(film_current)
                        if film_current in film_stocks
                        else 0,
                        key=f"regen_film_{shot_id_str}",
                    )

                    submitted = st.form_submit_button("🔁 Regenerate this shot")

                if submitted:
                    with st.spinner("Re-generating shot with updated parameters…"):
                        # Start from original, then apply UI changes
                        modified_fibo = copy.deepcopy(original_fibo)
                        modified_fibo.setdefault(
                            "photographic_characteristics", {}
                        )["camera_angle"] = new_angle
                        modified_fibo.setdefault("aesthetics", {})[
                            "mood_atmosphere"
                        ] = new_mood
                        modified_fibo["aesthetics"]["color_scheme"] = new_colors

                        # New pro controls written into FIBO
                        modified_fibo["hdr_mode"] = hdr_mode
                        modified_fibo["lighting_blueprint"] = lighting_blueprint
                        modified_fibo["film_stock"] = film_stock

                        # Generate a new image using the updated FIBO JSON
                        new_img_path = (
                            fibo_image_generator.generate_image_from_fibo_json(
                                modified_fibo
                            )
                        )

                    # ✅ Persist FIBO + image back into the storyboard model
                    # Update this shot's FIBO payload
                    fibo_payload["fibo_json"] = modified_fibo
                    shot_id_to_fibo[shot_id_str] = modified_fibo

                    # Update the generated_images entry for this shot
                    for img_meta in result["generated_images"]:
                        img_shot_id = str(
                            img_meta.get("shot_id", img_meta.get("shot_id", ""))
                        )
                        if img_shot_id == shot_id_str:
                            img_meta["image_path"] = new_img_path

                    # Also refresh in-memory lookup so downstream panels see the new image
                    shot_id_to_image_path[shot_id_str] = new_img_path

                    # Store updated storyboard back into session state so the next rerun reflects changes
                    st.session_state["storyboard"] = result

                    st.success("Shot re-generated!")

                    two_cols = st.columns(2)
                    with two_cols[0]:
                        st.image(
                            img_path,
                            use_container_width=True,
                            caption="Original",
                        )
                    with two_cols[1]:
                        st.image(
                            new_img_path,
                            use_container_width=True,
                            caption="Regenerated (edited JSON)",
                        )

                    with st.expander("Modified FIBO StructuredPrompt JSON"):
                        st.json(modified_fibo)

    # After building shot maps, expose frames to Gemini
    if shot_id_to_image_path:
        st.session_state["gemini_frames"] = shot_id_to_image_path

    # ========================
    # QC + review section
    # ========================
    st.subheader("✅ Quality Check & Review")
    st.markdown("**QC Report:**")
    if result["qc_report"]:
        st.json(result["qc_report"])
    else:
        st.write("All shots passed basic QC checks.")

    st.markdown("**Reviewer summary:**")
    st.json(result["review"])

    # ========================
    # NEW: Director’s Commentary
    # ========================
    st.markdown("---")
    st.subheader("🎙 Director’s Commentary")
    with st.expander("Show director-style explanation of visual choices", expanded=True):
        commentary = build_directors_commentary(result)
        st.write(commentary)

    # ========================
    # NEW: Continuity Inspector
    # ========================
    st.subheader("🔁 Continuity Inspector")
    st.caption(
        "Visual overview of camera, lighting, color and film-stock continuity across scenes and shots."
    )

    continuity_rows: List[Dict[str, Any]] = []
    for shot in shots:
        shot_id_str = str(shot.get("shot_id", "UNKNOWN"))
        scene_id = shot.get("scene", "?")
        fibo = shot_id_to_fibo.get(shot_id_str, {})
        pc = fibo.get("photographic_characteristics", {})
        aesth = fibo.get("aesthetics", {})
        env = shot.get("environment", {})

        hdr_mode = fibo.get("hdr_mode", "off")
        lighting_blueprint = fibo.get("lighting_blueprint", "default")
        film_stock = fibo.get("film_stock", "none")
        camera_angle = pc.get("camera_angle", "")

        dsl_summary = (
            f"S{scene_id}–{shot_id_str} | "
            f"angle={camera_angle or 'eye-level'} | "
            f"hdr={hdr_mode} | "
            f"light={lighting_blueprint} | "
            f"film={film_stock}"
        )

        continuity_rows.append(
            {
                "Scene": scene_id,
                "Shot": shot_id_str,
                "Camera angle": camera_angle,
                "Lens / FOV": pc.get("lens_focal_length", ""),
                "HDR mode": hdr_mode,
                "Lighting blueprint": lighting_blueprint,
                "Film stock": film_stock,
                "Mood": aesth.get("mood_atmosphere", ""),
                "Color scheme": aesth.get("color_scheme", ""),
                "Location": env.get("location", ""),
                "DSL summary": dsl_summary,
            }
        )

    if continuity_rows:
        st.dataframe(continuity_rows, hide_index=True)
    else:
        st.write("No continuity data available yet – generate a storyboard first.")

    # ========================
    # Gemini Character Continuity (vision)
    # ========================
    st.markdown("### 😁 Gemini Character Continuity (beta)")
    st.caption(
        "Use Gemini 2.5 Pro Vision to annotate character appearance, props, pose and emotion "
        "across all frames, so you can spot continuity breaks at a glance."
    )

    backend_base = st.session_state.get("backend_base", RENDER_BACKEND_BASE)

    frames_map: Dict[str, str] = st.session_state.get("gemini_frames", {})
    existing_ann: List[Dict[str, Any]] = st.session_state.get("gemini_annotations", [])

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("Let's Analyze Continuity!"):
            # We'll send the frame paths + shot IDs we know
            if not frames_map:
                st.warning(
                    "No frames found for Gemini analysis. Generate a storyboard first."
                )
            else:
                shot_ids = list(frames_map.keys())
                frame_paths = [frames_map[sid] for sid in shot_ids]

                url = f"{backend_base.rstrip('/')}/tools/gemini/character-continuity"
                payload = {
                    "image_paths": frame_paths,
                    "shot_ids": shot_ids,
                    "scene_prompts": [None] * len(frame_paths),
                }

                try:
                    resp = requests.post(url, json=payload, timeout=120)
                    if resp.status_code != 200:
                        st.error(
                            f"Gemini continuity endpoint failed ({resp.status_code}) for URL {url}: {resp.text}"
                        )
                    else:
                        data = resp.json()
                        anns = data.get("annotations", [])
                        st.session_state["gemini_annotations"] = anns
                        existing_ann = anns
                        st.success("Gemini continuity analysis complete! ✅")
                except Exception as exc:
                    st.error(f"Gemini continuity request failed: {exc}")

    if not existing_ann:
        st.info("Run continuity analysis to see annotations.")
    else:
        st.success("Gemini continuity analysis complete!")

        # ---- 1) Table view (Continuity Inspector) ----
        df_rows = []
        for ann in existing_ann:
            df_rows.append(
                {
                    "Shot": ann.get("shot_id") or "",
                    "Hair": ann.get("hair_color") or "",
                    "Clothing": ", ".join(ann.get("clothing") or []),
                    "Props": ", ".join(ann.get("props") or []),
                    "Expression": ann.get("expression") or "",
                    "Pose": ann.get("pose") or "",
                    "Continuity tags": ", ".join(ann.get("continuity_tags") or []),
                }
            )
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ---- 2) Annotated image preview with dropdown ----
        st.markdown("#### Annotated frame preview")

        # Build a list of shot labels that we can show
        shot_labels = [row["Shot"] or f"Shot {i+1}" for i, row in enumerate(df_rows)]
        # Map labels → annotation dict
        label_to_ann = {label: existing_ann[i] for i, label in enumerate(shot_labels)}

        # Default: first shot
        default_index = 0

        selected_label = st.selectbox(
            "Select a shot to preview:",
            options=shot_labels,
            index=default_index if shot_labels else 0,
        )

        selected_ann = label_to_ann.get(selected_label)
        selected_shot_id = selected_ann.get("shot_id") if selected_ann else None

        # Try to look up the frame path from our map
        frame_path = None
        if selected_shot_id and selected_shot_id in frames_map:
            frame_path = frames_map[selected_shot_id]
        elif shot_labels and not selected_shot_id:
            # fall back: match by order
            # (only if we didn't get a shot_id back from Gemini)
            idx = shot_labels.index(selected_label)
            # align index with frames_map order
            frame_path = list(frames_map.values())[idx] if frames_map else None

        if frame_path and os.path.exists(frame_path):
            try:
                img = Image.open(frame_path).convert("RGB")
                draw = ImageDraw.Draw(img)

                ann_dict = selected_ann or {}
                bbox = ann_dict.get("bounding_box") or ann_dict.get("face_box")
                if bbox:
                    x = bbox.get("x", 0)
                    y = bbox.get("y", 0)
                    w = bbox.get("width", 0)
                    h = bbox.get("height", 0)
                    # convert to bottom-right coords
                    x2, y2 = x + w, y + h

                    # primary face box
                    draw.rectangle([x, y, x2, y2], outline="red", width=4)

                st.image(
                    img,
                    caption=f"{selected_label} – annotated with Gemini character box",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Failed to render annotated frame: {exc}")
        else:
            st.warning(
                "Could not resolve frame path for this shot. "
                "Make sure `st.session_state['gemini_frames']` is set."
            )

        # ---- 3) Raw Gemini notes (optional) ----
        with st.expander("Raw Gemini notes per shot"):
            for ann in existing_ann:
                sid = ann.get("shot_id") or "Unknown shot"
                st.markdown(f"**{sid}**")
                st.markdown(ann.get("raw_model_notes") or "_(no notes)_")
                st.markdown("---")

    # -------------------------------------------------
    # Music Video export: mv_video_plan.json
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("🎵 Music Video Export")

    video_backend = st.session_state.get("video_backend", "svd")
    video_plan = build_video_plan(result, video_backend)
    json_bytes = json.dumps(video_plan, indent=2).encode("utf-8")

    st.download_button(
        "⬇️ Download mv_video_plan.json",
        data=json_bytes,
        file_name="mv_video_plan.json",
        mime="application/json",
        key="download_mv_plan",
    )
    st.caption(
        "Use this JSON in a Colab notebook or external pipeline to turn "
        "keyframes into music video clips using Stable Video Diffusion, LongCat, "
        "or any other backend."
    )

    # --- Render MV Button (calls external video backend via FastAPI) ---
    st.markdown("#### Or render the full MV using your video backend:")
    if st.button("🎬 Render Full Music Video", key="btn_render_mv"):
        with st.spinner(
            "Sending video plan to backend and rendering music video..."
        ):
            try:
                resp = requests.post(
                    f"{RENDER_BACKEND_BASE.rstrip('/')}/render-mv-json",
                    json={"plan": video_plan},
                    timeout=3600,
                )
                if resp.status_code != 200:
                    st.error(f"Backend error: {resp.status_code} {resp.text[:400]}")
                else:
                    mv_result = resp.json()
                    st.session_state["last_mv_result"] = mv_result

                    mv_url = mv_result.get("mv_url")
                    if mv_url:
                        # If backend returned a full URL, use it, otherwise prefix with backend base URL
                        if mv_url.startswith("http://") or mv_url.startswith(
                            "https://"
                        ):
                            full_url = mv_url
                        else:
                            full_url = f"{RENDER_BACKEND_BASE.rstrip('/')}{mv_url}"
                        st.session_state["rendered_mv_url"] = full_url
                        st.success("Music video rendered! Scroll down to preview.")
                    else:
                        st.warning("Backend did not return an 'mv_url' field.")
            except Exception as e:
                st.error(f"Error calling backend: {e}")

    # -------------------------------------------------
    # ComfyUI export – FIBO → node graph
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("🧩 ComfyUI Export")

    st.caption(
        "Export your current storyboard + FIBO JSON as a ComfyUI node graph so you can "
        "wire it into your own workflows (LoRA, ControlNet, etc.)."
    )

    comfy_graph = st.session_state.get("comfyui_graph")

    if st.button("📤 Export to ComfyUI graph", key="btn_export_comfyui"):
        with st.spinner("Building ComfyUI graph via FastAPI…"):
            try:
                resp = requests.post(
                    f"{RENDER_BACKEND_BASE.rstrip('/')}/tools/bria/export-comfyui",
                    json={
                        "shots": shots,
                        "fibo_payloads": fibo_payloads,
                    },
                    timeout=120,
                )
                if resp.status_code != 200:
                    st.error(
                        f"ComfyUI export failed: {resp.status_code} {resp.text[:400]}"
                    )
                else:
                    data = resp.json()
                    comfy_graph = data.get("graph", {})
                    st.session_state["comfyui_graph"] = comfy_graph
                    st.success("ComfyUI graph template built!")
            except Exception as e:
                st.error(f"Error calling /tools/bria/export-comfyui: {e}")
                comfy_graph = None

    if comfy_graph:
        graph_bytes = json.dumps(comfy_graph, indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download comfyui_graph.json",
            data=graph_bytes,
            file_name="comfyui_graph.json",
            mime="application/json",
            key="download_comfyui_graph",
        )
        with st.expander("Preview ComfyUI graph JSON"):
            st.json(comfy_graph)
    else:
        st.caption(
            "No ComfyUI graph yet. Click 'Export to ComfyUI graph' to generate one."
        )

    # -------------------------------------------------
    # Shot Asset Lab – RMBG, enhance, background variants
    # -------------------------------------------------
    st.markdown("---")
    st.subheader(
        "🧪 Shot Asset Lab – foreground cut-out, enhance, background variants & asset pack"
    )

    if not shot_id_to_image_path:
        st.info("Generate a storyboard first to use the Shot Asset Lab.")
        return

    default_shot_id = st.session_state.get(
        "asset_lab_shot_id", next(iter(shot_id_to_image_path.keys()))
    )
    selected_shot_id = st.selectbox(
        "Select a shot to edit",
        list(shot_id_to_image_path.keys()),
        index=list(shot_id_to_image_path.keys()).index(default_shot_id),
        key="asset_lab_shot_select",
    )
    selected_image_path = shot_id_to_image_path[selected_shot_id]

    col_preview, col_tools = st.columns([2, 3])

    with col_preview:
        if os.path.exists(selected_image_path):
            st.image(
                selected_image_path,
                use_container_width=True,
                caption=f"Shot {selected_shot_id} – original",
            )
        else:
            st.warning(f"Image not found: {selected_image_path}")

    with col_tools:
        st.markdown("**Foreground cut-out & enhancement (RMBG 2.0 + Enhance)**")

        if st.button("✂️ Remove background (RMBG 2.0)", key="btn_rmbg"):
            token = _ensure_bria_token()
            if token:
                try:
                    img_b64 = _image_file_to_base64(Path(selected_image_path))
                    with st.spinner("Calling Bria RMBG 2.0…"):
                        cutout_path = _bria_edit_call(
                            "remove_background", {"image": img_b64}
                        )
                    st.success("Background removed!")
                    st.session_state["asset_lab_cutout_path"] = cutout_path
                except Exception as e:
                    st.error(f"RMBG error: {e}")

        cutout_path = st.session_state.get("asset_lab_cutout_path")

        if cutout_path and os.path.exists(cutout_path):
            st.image(
                cutout_path,
                use_container_width=True,
                caption="Foreground cut-out (PNG with alpha)",
            )

            if st.button("📈 Enhance cut-out to 2MP", key="btn_enhance"):
                try:
                    img_b64 = _image_file_to_base64(Path(cutout_path))
                    with st.spinner("Calling Bria Enhance (2MP)…"):
                        enhanced_path = _bria_edit_call(
                            "enhance",
                            {"image": img_b64, "resolution": "2MP"},
                        )
                    st.success("Cut-out enhanced!")
                    st.session_state["asset_lab_enhanced_path"] = enhanced_path
                except Exception as e:
                    st.error(f"Enhance error: {e}")

        enhanced_path = st.session_state.get("asset_lab_enhanced_path")
        if enhanced_path and os.path.exists(enhanced_path):
            st.image(
                enhanced_path,
                use_container_width=True,
                caption="Enhanced foreground (2MP)",
            )

        st.markdown("**Bria Upscale (side-by-side)**")
        upscale_scale = st.selectbox(
            "Upscale factor",
            [2, 4],
            index=0,
            key="asset_lab_upscale_scale",
        )

        if st.button("🔍 Upscale this shot", key="btn_bria_upscale"):
            base_for_upscale = enhanced_path or selected_image_path
            if not base_for_upscale or not os.path.exists(base_for_upscale):
                st.error("No image available to upscale.")
            else:
                try:
                    # Call FastAPI local upscaler stub (no external GPU / Bria HTTP needed)
                    rel_path = (
                        str(Path(base_for_upscale).relative_to(ROOT))
                        if base_for_upscale.startswith(str(ROOT))
                        else base_for_upscale
                    )
                    with st.spinner("Calling /tools/bria/upscale-local…"):
                        resp = requests.post(
                            f"{RENDER_BACKEND_BASE.rstrip('/')}/tools/bria/upscale-local",
                            json={
                                "image_path": rel_path,
                                "scale": upscale_scale,
                            },
                            timeout=300,
                        )
                    if resp.status_code != 200:
                        st.error(
                            f"Upscale failed: {resp.status_code} {resp.text[:400]}"
                        )
                    else:
                        data = resp.json()
                        out_rel = data.get("output_path", rel_path)
                        upscaled_path = str((ROOT / out_rel).resolve())
                        st.session_state["asset_lab_upscaled_path"] = upscaled_path
                        st.success("Upscaled image generated!")
                except Exception as e:
                    st.error(f"Error calling upscale endpoint: {e}")

        upscaled_path = st.session_state.get("asset_lab_upscaled_path")
        if upscaled_path and os.path.exists(upscaled_path):
            side_cols = st.columns(2)
            with side_cols[0]:
                if os.path.exists(selected_image_path):
                    st.image(
                        selected_image_path,
                        use_container_width=True,
                        caption="Original",
                    )
            with side_cols[1]:
                st.image(
                    upscaled_path,
                    use_container_width=True,
                    caption=f"Upscaled ×{upscale_scale}",
                )

        st.markdown("**Background variants (Replace Background)**")
        bg_prompt = st.text_input(
            "Describe new background:",
            value="cinematic neon city street at night, soft rain, bokeh lights",
            key="bg_prompt_input",
        )
        num_variants = st.slider(
            "Number of variants", 1, 4, 2, key="bg_num_variants_slider"
        )

        if st.button("🎨 Generate background variants", key="btn_bg_variants"):
            base_for_variants = enhanced_path or cutout_path or selected_image_path
            if not base_for_variants:
                st.error("No image available for background replacement.")
            else:
                try:
                    img_b64 = _image_file_to_base64(Path(base_for_variants))
                    variant_paths: List[str] = []
                    with st.spinner("Calling Bria Replace Background…"):
                        for i in range(num_variants):
                            path = _bria_edit_call(
                                "replace_background",
                                {
                                    "image": img_b64,
                                    "prompt": bg_prompt,
                                    "mode": "high_control",
                                },
                            )
                            variant_paths.append(path)
                    st.success("Background variants generated!")
                    st.session_state["asset_lab_bg_variants"] = variant_paths
                except Exception as e:
                    st.error(f"Background replace error: {e}")

        variant_paths = st.session_state.get("asset_lab_bg_variants", [])
        if variant_paths:
            cols = st.columns(len(variant_paths))
            for col, vp in zip(cols, variant_paths):
                with col:
                    if os.path.exists(vp):
                        st.image(
                            vp,
                            use_container_width=True,
                            caption=Path(vp).name,
                        )

        # Per-shot asset pack download (ZIP with original, cut-out, enhanced, variants, metadata)
        if st.button("📦 Download shot asset pack (ZIP)", key="btn_zip_assets"):
            base_for_pack = enhanced_path or cutout_path or selected_image_path
            if not base_for_pack:
                st.error("No image available to pack.")
            else:
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    # Original
                    if os.path.exists(selected_image_path):
                        zf.write(
                            selected_image_path,
                            arcname="original_frame.png",
                        )
                    # Foreground (enhanced or raw cut-out)
                    if enhanced_path and os.path.exists(enhanced_path):
                        zf.write(
                            enhanced_path,
                            arcname="foreground_enhanced.png",
                        )
                    elif cutout_path and os.path.exists(cutout_path):
                        zf.write(
                            cutout_path,
                            arcname="foreground_cutout.png",
                        )

                    # Background variants
                    for i, vp in enumerate(variant_paths or []):
                        if os.path.exists(vp):
                            zf.write(
                                vp,
                                arcname=f"bg_variant_{i+1}.png",
                            )

                    # Simple metadata
                    meta = {
                        "shot_id": selected_shot_id,
                        "background_prompt": bg_prompt,
                    }
                    zf.writestr(
                        "metadata.json",
                        json.dumps(meta, indent=2),
                    )

                buf.seek(0)
                st.download_button(
                    "⬇️ Save shot_assets.zip",
                    data=buf,
                    file_name=f"shot_{selected_shot_id}_assets.zip",
                    mime="application/zip",
                    key="download_shot_zip",
                )

    # -------------------------------------------------
    # Full Music Video preview (if rendered externally)
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("🎬 Full Music Video Preview")

    mv_url = st.session_state.get("rendered_mv_url")
    if mv_url:
        st.video(mv_url)
        st.caption(
            "This final MV was rendered by the external image→video backend (e.g., SVD, LongCat, Pika, Runway, fal.ai)."
        )
    else:
        st.info(
            "No final music video yet. Click 'Render Full Music Video' above to generate one."
        )


if __name__ == "__main__":
    main()