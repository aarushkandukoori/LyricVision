"""Image generation — local Stable Diffusion when available, Pollinations API fallback."""

import asyncio
import base64
import io
import urllib.parse
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

from .prompts import PromptBundle

_pipe = None
_pipe_kind = None
_backend: str | None = None


def get_backend() -> str:
    global _backend
    if _backend is not None:
        return _backend
    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionPipeline  # noqa: F401

        _backend = "local"
    except ImportError:
        _backend = "remote"
    return _backend


def _load_local_pipeline():
    global _pipe, _pipe_kind
    if _pipe is not None:
        return _pipe, _pipe_kind

    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    _pipe, _pipe_kind = pipe, "sd15"
    return pipe, _pipe_kind


def _generate_local(bundle: PromptBundle, seed: int) -> Image.Image:
    import torch

    pipe, _ = _load_local_pipeline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    kwargs = {
        "prompt": bundle.prompt,
        "guidance_scale": bundle.guidance_scale,
        "num_inference_steps": bundle.num_inference_steps,
        "height": 512,
        "width": 512,
        "generator": generator,
    }
    if bundle.negative_prompt:
        kwargs["negative_prompt"] = bundle.negative_prompt

    return pipe(**kwargs).images[0]


async def _generate_remote(bundle: PromptBundle, seed: int) -> Image.Image:
    full_prompt = bundle.prompt
    if bundle.negative_prompt:
        avoid = ", ".join(bundle.negative_prompt.split(", ")[:6])
        full_prompt = f"{bundle.prompt}. Avoid: {avoid}"

    encoded = urllib.parse.quote(full_prompt[:900])
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=512&height=512&seed={seed}&nologo=true"
    )

    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
        for attempt in range(2):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                if attempt == 0:
                    await asyncio.sleep(1.5)
                    continue
                raise


def _placeholder_image(bundle: PromptBundle) -> Image.Image:
    """Fallback when remote generation fails."""
    palette = {
        "sadness": (30, 40, 90),
        "joy": (220, 160, 50),
        "anger": (140, 20, 30),
        "fear": (20, 50, 60),
        "neutral": (80, 80, 90),
        "confidence": (180, 140, 40),
    }
    bg = palette.get(bundle.primary_emotion, (50, 50, 70))
    img = Image.new("RGB", (512, 512), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small = font

    draw.text((20, 20), bundle.label, fill=(255, 255, 255), font=font)
    draw.text((20, 55), f"Emotion: {bundle.primary_emotion}", fill=(220, 220, 240), font=small)
    draw.text((20, 80), f"Genre: {bundle.primary_genre}", fill=(220, 220, 240), font=small)

    y = 120
    for line in _wrap(bundle.prompt, 55):
        draw.text((20, y), line, fill=(200, 200, 220), font=small)
        y += 18
        if y > 480:
            break
    return img


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        test = " ".join(current + [w])
        if len(test) > width:
            if current:
                lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return lines


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def generate_image(bundle: PromptBundle, seed: int = 42) -> tuple[Image.Image, str, str]:
    backend = get_backend()
    if backend == "local":
        try:
            img = _generate_local(bundle, seed)
            return img, image_to_base64(img), "local-sd15"
        except Exception:
            pass

    try:
        img = await _generate_remote(bundle, seed)
        return img, image_to_base64(img), "pollinations"
    except Exception:
        img = _placeholder_image(bundle)
        return img, image_to_base64(img), "placeholder"
