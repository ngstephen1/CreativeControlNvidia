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

import requests
import streamlit as st
from dotenv import load_dotenv

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

GENERATED_DIR = ROOT / "generated"
ASSETS_DIR = GENERATED_DIR / "assets"
GENERATED_DIR.mkdir(exist_ok=True, parents=True)
ASSETS_DIR.mkdir(exist_ok=True, parents=True)

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
    """
    Call a Bria v2 image-edit endpoint and return local path to downloaded image.
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


# -------------------------------------------------
# Multi-agent storyboard pipeline
# -------------------------------------------------
def run_full_pipeline(script_text: str) -> Dict[str, Any]:
    """
    Mirror of /full-pipeline-generate-images but as a plain function
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
        fibo_json = fibo_builder.sh_to_fibo_json(shot) if hasattr(
            fibo_builder, "sh_to_fibo_json"
        ) else fibo_builder.shot_to_fibo_json(shot)
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
    """
    Build a JSON-serializable plan that describes which storyboard frames
    should become which video clips. This is meant to be consumed by a
    Colab notebook or external service (e.g. Stable Video Diffusion, Pika,
    Runway) for image→video or text→video generation.
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
                "duration_sec": float(shot.get("duration_sec", 4.0)),
                "motion_style": shot.get("motion_style", "slow_cinematic_push_in"),
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

    st.title("🎬 Autonomous Studio Director")
    st.caption("JSON-native FIBO storyboard generator (Bria) + Music Video export")

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader("Script")
        default_script = (
            "Scene 1: Night rain on the city. A single violinist plays under a street lamp "
            "while cars pass in the distance.\n\n"
            "Scene 2: The next morning, the same violinist sits in a quiet cafe, "
            "looking out the window."
        )
        script_text = st.text_area(
            "Paste your scene description or multi-scene script:",
            value=default_script,
            height=220,
        )

        generate_button = st.button("🚀 Generate Storyboard", type="primary")

    with col_right:
        st.subheader("Run settings")
        st.markdown(
            "- Uses your local **multi-agent pipeline**\n"
            "- Calls Bria FIBO via **structured_prompt**\n"
            "- Saves frames to `generated/` folder\n"
        )

        # Video backend selection (for mv_video_plan)
        backend_label_map = {
            "Stable Video Diffusion (SVD – open-source, Colab-friendly)": "svd",
            "Pika (cinematic API)": "pika",
            "Runway Gen-2 (API)": "runway",
        }
        default_backend_label = "Stable Video Diffusion (SVD – open-source, Colab-friendly)"
        selected_backend_label = st.selectbox(
            "Video backend for image→video (used in export plan):",
            list(backend_label_map.keys()),
            index=list(backend_label_map.keys()).index(default_backend_label),
        )
        video_backend_value = backend_label_map[selected_backend_label]
        st.session_state["video_backend"] = video_backend_value

        if "last_run_summary" in st.session_state:
            st.markdown("**Last run:**")
            st.json(st.session_state["last_run_summary"])

        if BRIA_API_TOKEN:
            st.success("BRIA_API_TOKEN loaded.")
        else:
            st.warning("BRIA_API_TOKEN not found – generation will fail for edit tools.")

    # Run full pipeline when button clicked
    if generate_button:
        if not script_text.strip():
            st.error("Please enter a script first.")
            return

        with st.spinner("Generating storyboard with FIBO…"):
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

    # Map shot_id -> path for the Shot Asset Lab
    shot_id_to_image_path: Dict[str, str] = {}

    for shot, img_meta, fibo_payload in zip(shots, images, fibo_payloads):
        shot_id = shot.get("shot_id")
        scene_id = shot.get("scene")
        shot_id_to_image_path[shot_id] = img_meta["image_path"]

        st.markdown(f"### Scene {scene_id} – Shot {shot_id}")

        cols = st.columns([2, 3])

        # Left: original image
        with cols[0]:
            img_path = img_meta["image_path"]
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True, caption="Original frame")
            else:
                st.warning(f"Image not found: {img_path}")

            # Allow sending this shot into Shot Asset Lab
            if st.button(
                "🧪 Send to Shot Asset Lab",
                key=f"send_to_lab_{shot_id}",
            ):
                st.session_state["asset_lab_shot_id"] = shot_id
                st.session_state["asset_lab_image_path"] = img_path
                st.info(f"Shot {shot_id} sent to Shot Asset Lab (below).")

        # Right: metadata + JSON + controls
        with cols[1]:
            st.markdown("**Description**")
            st.write(shot.get("description", ""))

            st.markdown("**Camera intent**")
            st.write(shot.get("camera_intent", ""))

            st.markdown("**Environment**")
            st.json(shot.get("environment", {}))

            # 🎵 Music Video settings per shot (duration & motion style)
            with st.expander("🎵 Music Video Settings"):
                current_duration = float(shot.get("duration_sec", 4.0))
                current_motion = shot.get(
                    "motion_style", "slow_cinematic_push_in"
                )

                new_duration = st.slider(
                    "Shot duration (seconds)",
                    min_value=1.0,
                    max_value=12.0,
                    value=current_duration,
                    step=0.5,
                    key=f"dur_{shot_id}",
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
                    index=0,
                    key=f"motion_{shot_id}",
                )

                # Persist back into shot dict (so export + QC see updated values)
                shot["duration_sec"] = new_duration
                shot["motion_style"] = new_motion

                st.caption(
                    "These values will be embedded in mv_video_plan.json "
                    "for Colab / Pika / Runway."
                )

            with st.expander("FIBO StructuredPrompt JSON (original)"):
                st.json(fibo_payload["fibo_json"])

            # 🎛 Controllability panel – structured JSON tweaks
            with st.expander("🎛 Tweak & Regenerate this shot"):
                with st.form(f"regen_form_{shot_id}"):
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

                    new_angle = st.selectbox(
                        "Camera angle",
                        angle_options,
                        index=angle_index,
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
                        index=0,
                    )

                    new_colors = st.selectbox(
                        "Color scheme",
                        [
                            "cool blues and purples with warm highlights from artificial lights",
                            "warm orange and teal blockbuster palette",
                            "soft neutral daylight colors",
                            "high-contrast noir with deep shadows",
                        ],
                        index=0,
                    )

                    submitted = st.form_submit_button("🔁 Regenerate this shot")

                if submitted:
                    with st.spinner("Re-generating shot with updated parameters…"):
                        modified_fibo = copy.deepcopy(original_fibo)
                        modified_fibo.setdefault(
                            "photographic_characteristics", {}
                        )["camera_angle"] = new_angle
                        modified_fibo.setdefault("aesthetics", {})[
                            "mood_atmosphere"
                        ] = new_mood
                        modified_fibo["aesthetics"]["color_scheme"] = new_colors

                        new_img_path = (
                            fibo_image_generator.generate_image_from_fibo_json(
                                modified_fibo
                            )
                        )

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

    # QC + review section
    st.subheader("✅ Quality Check & Review")
    st.markdown("**QC Report:**")
    if result["qc_report"]:
        st.json(result["qc_report"])
    else:
        st.write("All shots passed basic QC checks.")

    st.markdown("**Reviewer summary:**")
    st.json(result["review"])

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
    )
    st.caption(
        "Use this JSON in a Colab notebook or external pipeline to turn "
        "keyframes into music video clips using Stable Video Diffusion, Pika, "
        "Runway, or any other backend."
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

        st.markdown("**Background variants (Replace Background)**")
        bg_prompt = st.text_input(
            "Describe new background:",
            value="cinematic neon city street at night, soft rain, bokeh lights",
        )
        num_variants = st.slider("Number of variants", 1, 4, 2)

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
                )


if __name__ == "__main__":
    main()