# 🎬 Autonomous Studio Director  
### _AI Storyboarding • Cinematic Shot Generation • Music Video Rendering_

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-UI-ff4b4b?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/FIBO-Structured%20Images-f39c12">
  <img src="https://img.shields.io/badge/LongCat-Video%20Gen-4a90e2">
  <img src="https://img.shields.io/badge/NVIDIA-Optimized-76b900?logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3.10+-blue">
</p>

---

## 🚀 Overview  
**Autonomous Studio Director** converts plain text scripts into cinematic storyboards and fully rendered music videos.  
Powered by **multi-agent reasoning**, **FIBO JSON controllability**, and **parallel video rendering**, it offers an end‑to‑end creative pipeline:

- Scene + shot breakdown using intelligent agents  
- FIBO‑structured JSON for professional‑grade controllability  
- Bria‑powered keyframe generation  
- LongCat (fal.ai) async video rendering  
- Interactive Streamlit editor for creative iteration  

This project targets film makers, creators, and AI‑powered production workflows.

---

## ✨ Core Features  
- 🎭 **Multi‑Agent Script → Storyboard**  
- 🖼️ **High‑control Image Generation** (camera, lighting, composition, palette, HDR)  
- 🎬 **Music Video Renderer** with automatic shot stitching  
- 🧪 **Shot Asset Lab** (RMBG, enhancements, background swaps)  
- 🚀 **Async Parallel Rendering** for speed + cost control  
- 📦 **One‑click Asset Export**  

---

## 🧰 Tech Stack  
- **UI:** Streamlit  
- **Backend:** FastAPI  
- **AI Models:** BRIA FIBO, ControlNet‑ready architecture  
- **Video:** LongCat (fal.ai)  
- **Storage:** `/generated` asset directory  

---

## 🔧 Quick Setup  

Create a `.env` file:

```
BRIA_API_TOKEN=your_bria_key  
FAL_KEY=your_fal_api_key  
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
Designed for the **FIBO × NVIDIA × Fal.ai Hackathon**, showcasing:  
- High controllability (camera, lighting, pose, palettes, HDR modes)  
- Multi‑agent creative direction  
- Real cinematic production workflow simulation  

---

## ❤️ Credits  
Built with the support of **Bria AI**, **NVIDIA**, **Fal.ai**, and the open‑source community.

---

## 📸 Demo Preview  
Here’s a quick look at what the Autonomous Studio Director produces:

- **Storyboard images** generated with professional camera + lighting control  
- **FIBO JSON blocks** showing structured cinematic intent  
- **Stitched music‑video clips** rendered through async LongCat pipelines  

*(Add your example images or GIFs to this section when available.)*

---

## 🧱 System Architecture (High‑Level)

```
User Script → Multi‑Agent Parser → FIBO JSON Builder  
        → Image Generator (BRIA) → Keyframes  
        → Parallel Video Engine (LongCat) → Final MV
```

- **Agents**: handle scene splitting, camera intention, environment mapping  
- **FIBO Builder**: produces HDR‑ready, controllable JSON  
- **Backend API**: orchestrates job dispatch + asset tracking  
- **Streamlit UI**: allows per‑shot refinement and interactive editing  

---

## 🎛 Advanced Controls Supported  

### Camera  
- Angle, lens, depth of field  
- Motion intent (dolly, pan, push‑in)  

### Lighting  
- Three‑point lighting  
- Noir hard‑shadows  
- Sunset/warm keylight  
- Neon reflections  

### Film Stocks  
- Kodak 5219  
- Fuji Eterna  
- Custom LUT presets  

### Composition  
- Golden ratio  
- Center‑weighted portrait  
- Wide establishing frames  

---

## 🗺️ Feature Roadmap  

### Phase 1 — Completed  
- Storyboard generator  
- HDR + Controllability layers  
- Parallel video stitching  
- Continuity inspector  

### Phase 2 — In Progress  
- BRIA ControlNet (pose, depth, canny, colorgrid)  
- Inpainting workflow  
- Editable masks per shot  

### Phase 3 — Planned  
- Audio‑synchronized shot timing  
- Beat detection → automatic pacing  
- Direct export to Premiere Pro / Resolve  
- Multi‑character continuity tracking  

---

## 📂 Project Structure Overview  

```
CreativeControlNvidia/
│
├── app/
│   ├── api.py               # FastAPI backend
│   ├── schemas.py           # Request/response models
│   └── utils/               # Helpers
│
├── fibo/
│   ├── image_generator_bria.py
│   ├── fibo_builder.py
│   └── presets/             # Camera/lighting/palette presets
│
├── video/
│   └── video_backend.py     # Async LongCat rendering
│
├── ui/
│   └── storyboard_app.py    # Streamlit front‑end
│
├── generated/               # Output images + videos
└── README.md
```

---

## 🤝 Contributing  

Pull requests are welcome!  
If you’d like to help build more cinematic controls (lens metadata, shot composition AI, or BRIA ControlNet integrations), feel free to open an issue.

---

## 🌟 If You Like This Project  
Please ⭐ the repo — it helps support ongoing development!