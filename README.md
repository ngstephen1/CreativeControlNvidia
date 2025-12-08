# 🎬 Autonomous Studio Director x FIBO Hackathon Sponsored by Fal.ai, NVIDIA and BRIA
_AI Storyboarding, Cinematic Shot Generation, and Music Video Rendering_

<p align="center">
  <img src="https://img.shields.io/badge/framework-Streamlit-ff4b4b?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/video-LongCat%20Video-4a90e2">
  <img src="https://img.shields.io/badge/storyboard-Bria%20FIBO-f39c12">
  <img src="https://img.shields.io/badge/agents-Multi--Agent%20Pipeline-9b59b6">
  <img src="https://img.shields.io/badge/lang-Python%203.10+-blue">
</p>

---

## 🚀 Overview
**Autonomous Studio Director** turns raw script text into a full cinematic pipeline:

- Multi-agent shot reasoning (Creative Director → Cinematography → Continuity → QC → Reviewer)  
- FIBO-structured JSON for every shot  
- Keyframe generation using Bria  
- Music-video rendering via LongCat (fal.ai)  
- Interactive Streamlit UI  
- Full Shot Asset Lab (RMBG, Enhance, background variants, ZIP export)

Everything runs locally except external image/video APIs.

---

## ✨ Features
- 🎭 Multi-agent cinematic breakdown  
- 🖼️ FIBO JSON → storyboard keyframes  
- 🎞️ One-click music-video generation  
- 🧪 Shot Asset Lab (cut-out, enhance, replace background)  
- 📦 ZIP asset export  
- ⚡ Ready for async parallel LongCat rendering  
- 📊 Progress feedback coming soon

---

## 🧰 Tech Stack
- **UI:** Streamlit  
- **Backend:** FastAPI  
- **Agents:** Python  
- **Images:** Bria FIBO  
- **Video:** LongCat (fal.ai)  
- **Storage:** `/generated` directory  

---

## 🔧 Environment Setup
Create a `.env`:

```ini
BRIA_API_TOKEN=your_bria_key
FAL_KEY=your_fal_api_key
RENDER_BACKEND_BASE=http://localhost:8000