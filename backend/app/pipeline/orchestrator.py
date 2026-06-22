"""Run all five ArtML pipeline versions on input lyrics."""

import asyncio

from .image_gen import generate_image, get_backend
from .prompts import PromptBundle, build_all_prompts


async def _generate_one(bundle: PromptBundle, seed: int) -> dict:
    img, b64, img_backend = await generate_image(bundle, seed=seed)
    return {
        "version": bundle.version,
        "label": bundle.label,
        "description": bundle.description,
        "prompt": bundle.prompt,
        "negative_prompt": bundle.negative_prompt,
        "keywords": bundle.keywords,
        "emotions": [{"label": e, "score": round(s, 3)} for e, s in bundle.emotions],
        "primary_emotion": bundle.primary_emotion,
        "primary_genre": bundle.primary_genre,
        "guidance_scale": bundle.guidance_scale,
        "num_inference_steps": bundle.num_inference_steps,
        "image_base64": b64,
        "image_backend": img_backend,
    }


async def run_all_pipelines(lyrics: str, seed: int = 42) -> dict:
    bundles = build_all_prompts(lyrics)
    results = await asyncio.gather(*[_generate_one(b, seed) for b in bundles])
    # Keep V1–V5 order
    order = {"v1": 0, "v2": 1, "v3": 2, "v4": 3, "v5": 4}
    results = sorted(results, key=lambda r: order[r["version"]])

    return {
        "results": results,
        "seed": seed,
        "nlp_backend": get_backend(),
    }
