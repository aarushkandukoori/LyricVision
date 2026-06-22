"""LyricVision API — compare ArtML pipeline versions V1–V5."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.pipeline.nlp_setup import ensure_nltk
from app.pipeline.orchestrator import run_all_pipelines

SAMPLE_LYRICS = """When the night falls and shadows creep
I find myself lost in memories deep
Every word you said, every tear I've cried
In this darkness, I search for the light inside
Feel the weight of the world on my shoulders now
But I'll rise again, I'll figure it out somehow"""


class GenerateRequest(BaseModel):
    lyrics: str = Field(..., min_length=10, max_length=8000)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_nltk()
    yield


app = FastAPI(
    title="LyricVision",
    description="Compare ArtML NLP-to-image pipeline versions V1–V5",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    from app.pipeline.image_gen import get_backend

    return {"status": "ok", "image_backend": get_backend()}


@app.get("/api/sample")
async def sample():
    return {"lyrics": SAMPLE_LYRICS}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    lyrics = req.lyrics.strip()
    if not lyrics:
        raise HTTPException(status_code=400, detail="Lyrics cannot be empty")
    try:
        return await run_all_pipelines(lyrics, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


frontend_dist = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
