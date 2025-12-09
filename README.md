# 🎬 Autonomous Studio Director  
### _AI Storyboarding • Cinematic Shot Generation • Music Video Rendering_

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/FIBO-AI%20Images-f39c12">
  <img src="https://img.shields.io/badge/LongCat-Video%20Gen-4a90e2">
  <img src="https://img.shields.io/badge/NVIDIA-Optimized-76b900?logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Google_Gemini-2.5_Pro-4285F4?logo=google&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
</p>

---

## 🚀 Overview  
**Autonomous Studio Director** converts plain text scripts into cinematic storyboards and fully rendered music videos using a multi‑agent filmmaking pipeline, FIBO JSON controllability, and BRIA + LongCat model integrations.

It features:

- Multi‑stage agent pipeline (Director → Cinematography → Continuity → QC)  
- FIBO controllability for lens, lighting, composition, and film‑stock style  
- BRIA image generation, enhancement, and background removal  
- Experimental BRIA ControlNet pipelines tested on Google Colab  
- Async LongCat (fal.ai) video generation  
- Streamlit UI for shot‑by‑shot refinement

This system accelerates pre‑production for filmmakers, creators, and AI‑first studios.

---

## ✨ Core Features  
- 🎭 **Multi‑Agent Script Breakdown → Storyboard**  
- 🖼️ **High‑Control FIBO Image Generation (camera, HDR, lighting, palette)**  
- 🎬 **Automated Music‑Video Renderer**  
- 🧪 **Shot Asset Lab** → RMBG, upscale, enhance  
- 🛠️ **Bria ControlNet Builder** (pose, canny, depth, colorgrid)  
- 📦 **Export to ComfyUI Graph**  
- 📐 **Continuity Inspector** powered by Gemini 2.5 Pro  
- 🚀 **Async Parallel Rendering (LongCat)**  

---

## 🧰 Tech Stack  
- **UI:** Streamlit  
- **Backend:** FastAPI  
- **Image Models:**  
  - **BRIA‑3.2** (HF gated model)  
  - **BRIA‑3.2‑ControlNet‑Union**  
  - **Custom BRIA pipelines:**  
    - `pipeline_bria.py`  
    - `pipeline_bria_controlnet.py`  
    - `controlnet_bria.py`  
    - `transformer_bria.py`  
  - **Tested via Google Colab (GPU Runtime)**  
- **Video:** LongCat (fal.ai)  
- **Continuity AI:** Gemini 2.5 Pro (bounding boxes + traits)  
- **Storage:** `/generated` asset directory  

---

## 🔬 Hugging Face Models Tested on Google Colab  

The following **BRIA models and pipelines** were successfully downloaded, imported, and partially executed on GPU runtimes:

### ✅ **BRIA‑3.2**  
```
repo_id="briaai/BRIA-3.2"
revision="pre_diffusers_support"
```

### ✅ **BRIA‑3.2‑ControlNet‑Union**  
Tested with custom loader:
- Canny condition  
- Depth condition  
- ColorGrid condition  
- Pose (OpenPose) condition  

### ✅ Custom BRIA Transformer + Pipeline Modules  
Loaded manually in strict order:

1. `bria_utils.py`  
2. `transformer_bria.py`  
3. `controlnet_bria.py`  
4. `pipeline_bria.py`  
5. `pipeline_bria_controlnet.py`

These modules were dynamically patched to resolve:
- relative imports  
- module path injection  
- missing safetensors fallbacks  
- dtype compatibility

We also validated:
- image resizing logic (ratio‑constrained)  
- inference using BF16  
- fallback to unsafe serialization when needed

All debugging steps are documented in the repo's issues.

---

## 🔧 Quick Setup  

Create a `.env` file:

```
BRIA_API_TOKEN=your_bria_key  
FAL_KEY=your_fal_api_key  
GEMINI_API_KEY=your_gemini_key
RENDER_BACKEND_BASE=http://localhost:8000
```

Run the backend:

```
uvicorn app.api:app --reload
```

Run the UI:

```
streamlit run ui/storyboard_app.py
```

---

## 🏆 Hackathon Focus  
Designed for the **FIBO × NVIDIA × Fal.ai Hackathon**, demonstrating:  
- Advanced cinematic controllability  
- High‑fidelity BRIA image generation  
- Multi‑character continuity using Gemini Pro  
- Modular rendering backend (LongCat)  
- Export‑ready workflow (ComfyUI)

---

## ❤️ Credits  
Built with support from:
- **Bria AI** (FIBO, Upscale, RMBG, ControlNet experiments)  
- **NVIDIA** (GPU runtimes + optimization study)  
- **Fal.ai** (LongCat async video gen)  
- **Google Colab** (BRIA pipeline testing environment)  

---

## 📸 Demo Preview  
Showcase your pipeline:

- Storyboard frames  
- Annotated continuity frames (bounding boxes + traits)  
- Upscaled + enhanced versions  
- Final rendered music video clips

_Add GIFs and screenshots when ready._

---

## 🧱 System Architecture (High‑Level)

```
User Script → Multi‑Agent Parser  
           → FIBO JSON Builder  
           → BRIA Image Generator / ControlNet  
           → Keyframes  
           → LongCat Parallel Video Engine  
           → Final MV Output
```

---

## 🎛 Advanced Controls  

### Camera  
- Angle, lens length, DOF, perspective  
- Dolly / pan / crane motion cues  

### Lighting  
- Natural HDR  
- Film noir hard‑shadow  
- Neon + reflective surfaces  

### Film Stock  
- Kodak 5219  
- Fuji Eterna  
- Custom LUT palettes  

### Composition  
- Golden ratio  
- Leading lines  
- Symmetry / Center‑weighted portrait  

---

## 🗺️ Feature Roadmap  

### Phase 1 — Completed  
- Storyboard generator  
- HDR + controllability layers  
- Parallel video stitching  
- Continuity inspector (Gemini)

### Phase 2 — In Progress  
- BRIA ControlNet integration (pose, depth, canny)  
- Local inpainting workflow  
- User‑editable masks  

### Phase 3 — Planned  
- Audio‑sync & beat‑driven pacing  
- Timeline editor  
- Premiere Pro / Resolve export  
- Multi‑character continuity over long videos  

---

## 📂 Project Structure Overview  

```
CreativeControlNvidia/
│
├── app/
│   ├── api.py
│   ├── schemas.py
│   └── utils/
│
├── fibo/
│   ├── image_generator_bria.py
│   ├── fibo_builder.py
│   └── presets/
│
├── gemini/
│   └── character_annotator.py
│
├── video/
│   └── video_backend.py
│
├── ui/
│   └── storyboard_app.py
│
├── generated/
└── README.md
```

---

## 🤝 Contributing  
Pull requests are welcome!  
Help us extend cinematic control, improve FIBO schemas, or optimize BRIA inference.
