"""Prompt builders V1–V5 from the ArtML pipeline iterations."""

from dataclasses import dataclass

from .emotions import classify_emotions_simple, classify_emotions_v5
from .genre import primary_genre
from .keywords import extract_keywords_v1_v4, extract_keywords_v5

EMOTION_VISUAL = {
    "sadness": {
        "palette": "deep cobalt blue and royal purple, indigo highlights",
        "lighting": "soft rim lighting, single warm light source breaking through fog",
        "scene_motif": "foggy park at dusk, lone figure under a streetlamp",
        "mood_words": "melancholic, intimate, quiet",
    },
    "fear": {
        "palette": "desaturated teal and charcoal, deep navy shadows",
        "lighting": "low-key lighting, harsh single key light, deep shadows",
        "scene_motif": "figure looking up toward storm clouds, wind in their hair",
        "mood_words": "tense, restrained, atmospheric",
    },
    "joy": {
        "palette": "warm golden yellow, coral pink, sky blue, vibrant orange",
        "lighting": "golden hour sunlight, lens flares, soft bokeh",
        "scene_motif": "figure mid-leap with confetti, open summer sky",
        "mood_words": "euphoric, weightless, radiant",
    },
    "anger": {
        "palette": "crimson red and inky black, hot ember orange",
        "lighting": "hard backlight, sparks, harsh contrast",
        "scene_motif": "figure facing camera, fire smoldering behind them",
        "mood_words": "fierce, kinetic, raw",
    },
    "surprise": {
        "palette": "electric cyan, hot magenta, pearl white",
        "lighting": "sudden flash, high-key, blown highlights",
        "scene_motif": "figure with eyes wide, light bursting from behind them",
        "mood_words": "electric, sudden, dazzling",
    },
    "disgust": {
        "palette": "sickly green and bruise purple",
        "lighting": "fluorescent overhead light, cold shadows",
        "scene_motif": "figure turning away in dim alley",
        "mood_words": "unsettled, abrasive, claustrophobic",
    },
    "neutral": {
        "palette": "muted earth tones, soft sage and dusty rose",
        "lighting": "overcast diffused daylight",
        "scene_motif": "figure standing still, looking off-frame",
        "mood_words": "contemplative, still, observational",
    },
    "confidence": {
        "palette": "rich gold, jet black, deep burgundy",
        "lighting": "dramatic top light, gold hour through skyscrapers",
        "scene_motif": "figure on a rooftop at sunset, city skyline behind",
        "mood_words": "commanding, regal, kinetic",
    },
}

GENRE_STYLE_ANCHOR = {
    "sad": "in the style of a 2000s indie album cover, 35mm film grain, cinematic photograph",
    "pop": "contemporary pop album art, glossy editorial photograph, magazine cover lighting",
    "rap": "1990s—2010s hip-hop album cover, editorial fashion photograph, hard light",
    "rock": "classic rock album cover, gritty 35mm film, high contrast",
    "electronic": "modern electronic album cover, synthwave aesthetic, neon-lit photograph",
    "folk": "folk album cover, natural light, large format photograph",
}

GENRE_GUIDANCE_SCALE = {
    "sad": 8.5, "pop": 6.5, "rap": 7.5,
    "rock": 7.5, "electronic": 7.0, "folk": 7.5,
}

UNIVERSAL_NEGATIVE = (
    "text, watermark, words, letters, typography, caption, subtitle, signature, "
    "logo, ugly, deformed, disfigured, low quality, blurry, jpeg artifacts, "
    "oversaturated, generic, stock photo, brown, tan, beige, sepia, dull, "
    "hallway, corridor, office building, conference room, parking lot, "
    "bad anatomy, extra limbs, malformed, low resolution"
)

VERSION_META = {
    "v1": {
        "label": "V1 — Original",
        "description": "Generic prompt from main.ipynb — basic emotion + keyword list",
        "steps": 15,
        "guidance_scale": 7.5,
    },
    "v2": {
        "label": "V2 — Enhanced Style",
        "description": "Adds cinematic style direction and high-contrast framing",
        "steps": 30,
        "guidance_scale": 7.5,
    },
    "v3": {
        "label": "V3 — Emotion Palette",
        "description": "Emotion-weighted color mapping and visual metaphors",
        "steps": 30,
        "guidance_scale": 7.5,
    },
    "v4": {
        "label": "V4 — Narrative",
        "description": "Story-driven prompt with cinematic composition language",
        "steps": 30,
        "guidance_scale": 7.5,
    },
    "v5": {
        "label": "V5 — Full Pipeline",
        "description": "Genre-aware, section-weighted emotions, visual metaphors, negative prompt",
        "steps": 30,
        "guidance_scale": None,
    },
}


@dataclass
class PromptBundle:
    version: str
    prompt: str
    negative_prompt: str
    guidance_scale: float
    num_inference_steps: int
    primary_emotion: str
    primary_genre: str
    keywords: list[str]
    emotions: list[tuple[str, float]]
    concept_string: str
    label: str
    description: str


def build_prompt_v1(emotions: list[tuple[str, float]], keywords: list[str]) -> str:
    top_emotions = [e[0] for e in emotions[:3]]
    return f"An album cover art that conveys {', '.join(top_emotions)} emotions featuring {', '.join(keywords[:5])}"


def build_prompt_v2(emotions: list[tuple[str, float]], keywords: list[str]) -> str:
    top_emotions = [e[0] for e in emotions[:3]]
    return (
        f"A dark, moody album cover art conveying {', '.join(top_emotions)} emotions. "
        f"Featured imagery should include: {', '.join(keywords[:5])}. "
        "Style: dramatic, cinematic, high contrast."
    )


def build_prompt_v3(emotions: list[tuple[str, float]], keywords: list[str]) -> str:
    primary = emotions[0][0] if emotions else "emotional"
    secondary = [e[0] for e in emotions[1:3]] if len(emotions) > 1 else []
    emotion_map = {
        "sadness": "deep blues and purples",
        "fear": "dark shadows and fog",
        "joy": "bright warm colors and light",
        "anger": "reds and intense contrast",
        "neutral": "muted tones",
    }
    palette = emotion_map.get(primary, "atmospheric")
    sec = ", ".join(secondary)
    return (
        f"Album cover: {primary.upper()} and {sec}. "
        f"Visual palette: {palette}. Key elements: {', '.join(keywords[:5])}. "
        "Professional music artwork, high quality."
    )


def build_prompt_v4(emotions: list[tuple[str, float]], keywords: list[str]) -> str:
    narratives = {
        "sadness": "An introspective journey through melancholy",
        "fear": "A tense, unsettling emotional landscape",
        "joy": "A celebration of light and euphoria",
        "anger": "Raw intensity and passionate conflict",
        "neutral": "A contemplative scene",
    }
    primary = emotions[0][0] if emotions else "emotion"
    narrative = narratives.get(primary, "An emotional journey")
    return (
        f"{narrative}. Incorporates: {', '.join(keywords[:5])}. "
        "Professional album cover design, cinematic lighting, artfully composed."
    )


def build_prompt_v5(lyrics: str) -> PromptBundle:
    genre, _ = primary_genre(lyrics)
    keywords = extract_keywords_v5(lyrics, genre, top_k=8)
    emotions = classify_emotions_v5(lyrics, genre)
    primary = emotions[0][0] if emotions else "neutral"
    visual = EMOTION_VISUAL.get(primary, EMOTION_VISUAL["neutral"])

    metaphors = [k for k in keywords if " " in k]
    if metaphors:
        scene = ", ".join(metaphors[:3]) + ", " + visual["scene_motif"]
    else:
        scene = f"{visual['scene_motif']}, including {', '.join(keywords[:4])}"

    prompt = (
        f"{visual['palette']}, {visual['lighting']}, "
        "subject fills 60% of frame, medium shot, centered composition, eye-level, "
        f"{scene}, "
        f"{visual['mood_words']}, "
        f"{GENRE_STYLE_ANCHOR.get(genre, '')}, square album cover, 1:1 aspect, highly detailed"
    )

    concept = (
        f"A {visual['mood_words']} {genre} album cover featuring "
        f"{', '.join(keywords[:5])} in {visual['palette']}."
    )

    meta = VERSION_META["v5"]
    return PromptBundle(
        version="v5",
        prompt=prompt,
        negative_prompt=UNIVERSAL_NEGATIVE,
        guidance_scale=GENRE_GUIDANCE_SCALE.get(genre, 7.5),
        num_inference_steps=meta["steps"],
        primary_emotion=primary,
        primary_genre=genre,
        keywords=keywords,
        emotions=emotions[:4],
        concept_string=concept,
        label=meta["label"],
        description=meta["description"],
    )


def build_prompt_v1_v4(lyrics: str, version: str) -> PromptBundle:
    keywords = extract_keywords_v1_v4(lyrics)
    emotions = classify_emotions_simple(lyrics)
    primary = emotions[0][0] if emotions else "neutral"
    genre, _ = primary_genre(lyrics)

    builders = {
        "v1": build_prompt_v1,
        "v2": build_prompt_v2,
        "v3": build_prompt_v3,
        "v4": build_prompt_v4,
    }
    prompt = builders[version](emotions, keywords)
    meta = VERSION_META[version]

    return PromptBundle(
        version=version,
        prompt=prompt,
        negative_prompt="",
        guidance_scale=meta["guidance_scale"],
        num_inference_steps=meta["steps"],
        primary_emotion=primary,
        primary_genre=genre,
        keywords=keywords,
        emotions=emotions[:4],
        concept_string=prompt,
        label=meta["label"],
        description=meta["description"],
    )


def build_all_prompts(lyrics: str) -> list[PromptBundle]:
    bundles = [build_prompt_v1_v4(lyrics, v) for v in ("v1", "v2", "v3", "v4")]
    bundles.append(build_prompt_v5(lyrics))
    return bundles
