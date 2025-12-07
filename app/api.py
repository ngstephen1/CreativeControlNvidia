# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from agents.creative_director import CreativeDirectorAgent

app = FastAPI(title="Autonomous Studio Director API")

creative_director = CreativeDirectorAgent()

class ScriptRequest(BaseModel):
    script_text: str

@app.post("/script-to-shots")
def script_to_shots(req: ScriptRequest):
    shots = creative_director.script_to_shots(req.script_text)
    return {"shots": shots}