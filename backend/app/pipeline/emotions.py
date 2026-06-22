"""Emotion classification — simple (V1–V4) and section-weighted (V5)."""

from collections import defaultdict

from .genre import GENRE_LEXICON, primary_genre
from .preprocess import preprocess_v5, split_sections

_emotion_pipe = None
_emotion_pipe_failed = False

EMOTION_KEYWORD_SIGNALS = {
    "sadness": {"sad", "cry", "tears", "lost", "dark", "pain", "alone", "memories", "shadow", "fog", "broken", "empty", "gone", "wound", "drowning"},
    "joy": {"joy", "happy", "shine", "bright", "dance", "alive", "thrive", "golden", "free", "love", "party", "confetti", "sun"},
    "anger": {"anger", "fight", "fire", "storm", "rebel", "blood", "fierce", "raw", "hate"},
    "fear": {"fear", "shadow", "dark", "creep", "tense", "storm", "wind"},
    "surprise": {"surprise", "electric", "sudden", "flash", "wide"},
    "neutral": {"still", "quiet", "contemplative", "observe"},
}

SECTION_WEIGHTS = {
    "chorus": 2.5,
    "pre-chorus": 1.2,
    "bridge": 1.5,
    "verse": 1.0,
    "verse 1": 1.0,
    "verse 2": 1.0,
    "intro": 0.5,
    "outro": 0.5,
}

GENRE_EMOTION_BIAS = {
    "rap": {
        "fear": 0.3, "sadness": 0.5, "joy": 1.2, "anger": 1.4, "surprise": 1.0,
        "disgust": 0.6, "neutral": 1.0, "confidence": 1.8,
    },
    "pop": {
        "fear": 0.6, "sadness": 0.5, "joy": 1.5, "anger": 0.5, "surprise": 1.1,
        "disgust": 0.4, "neutral": 0.8,
    },
    "sad": {
        "fear": 1.0, "sadness": 1.3, "joy": 0.7, "anger": 0.7, "surprise": 0.8,
        "disgust": 0.8, "neutral": 0.9,
    },
    "rock": {"anger": 1.4, "joy": 1.0, "fear": 0.8},
    "electronic": {"surprise": 1.3, "joy": 1.1, "neutral": 1.1},
    "folk": {"sadness": 1.1, "joy": 1.0, "neutral": 1.2},
}


def _keyword_emotions(lyrics: str) -> list[tuple[str, float]]:
    import re

    tokens = set(re.findall(r"[a-z']+", lyrics.lower()))
    scores = {e: 0.1 for e in EMOTION_KEYWORD_SIGNALS}
    for emotion, words in EMOTION_KEYWORD_SIGNALS.items():
        for w in words:
            if w in tokens:
                scores[emotion] += 1.0
    total = sum(scores.values()) or 1.0
    return sorted(((e, s / total) for e, s in scores.items()), key=lambda x: x[1], reverse=True)


def _get_emotion_pipe():
    global _emotion_pipe, _emotion_pipe_failed
    if _emotion_pipe_failed:
        return None
    if _emotion_pipe is not None:
        return _emotion_pipe
    try:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        try:
            _emotion_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                device=device,
            )
        except TypeError:
            _emotion_pipe = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=device,
            )
    except Exception:
        _emotion_pipe_failed = True
        return None
    return _emotion_pipe


def _normalize_scores(raw_output) -> list[dict]:
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
        return raw_output[0]
    if isinstance(raw_output, list):
        return raw_output
    return [raw_output]


def classify_emotions_simple(lyrics: str) -> list[tuple[str, float]]:
    """Single-pass emotion classification for V1–V4."""
    pipe = _get_emotion_pipe()
    if pipe is None:
        return _keyword_emotions(lyrics)
    try:
        out = pipe(lyrics[:512])
        scores = _normalize_scores(out)
        ranked = sorted(scores, key=lambda x: x["score"], reverse=True)
        return [(e["label"], float(e["score"])) for e in ranked]
    except Exception:
        return _keyword_emotions(lyrics)


def classify_emotions_v5(lyrics: str, genre: str) -> list[tuple[str, float]]:
    """Section-weighted, genre-biased emotion classification for V5."""
    pipe = _get_emotion_pipe()
    if pipe is None:
        raw = dict(_keyword_emotions(lyrics))
        bias = GENRE_EMOTION_BIAS.get(genre, {})
        adjusted = {k: v * bias.get(k, 1.0) for k, v in raw.items()}
        if genre == "rap":
            rap_signal = sum(1 for t in preprocess_v5(lyrics) if t in GENRE_LEXICON["rap"])
            adjusted["confidence"] = min(1.0, 0.05 + 0.04 * rap_signal) * bias.get("confidence", 1.0)
        total = sum(adjusted.values()) or 1.0
        return sorted(((k, v / total) for k, v in adjusted.items()), key=lambda x: x[1], reverse=True)

    sections = split_sections(lyrics)
    accum: dict[str, float] = defaultdict(float)
    total_weight = 0.0

    for label, body in sections:
        weight = SECTION_WEIGHTS.get(label, 1.0)
        if not body.strip():
            continue
        try:
            out = pipe(body[:512])
        except Exception:
            continue
        scores = _normalize_scores(out)
        for s in scores:
            accum[s["label"]] += weight * s["score"]
        total_weight += weight

    if total_weight == 0:
        return classify_emotions_simple(lyrics)

    raw = {k: v / total_weight for k, v in accum.items()}
    bias = GENRE_EMOTION_BIAS.get(genre, {})
    adjusted = {k: v * bias.get(k, 1.0) for k, v in raw.items()}

    if genre == "rap":
        rap_signal = sum(1 for t in preprocess_v5(lyrics) if t in GENRE_LEXICON["rap"])
        adjusted["confidence"] = min(1.0, 0.05 + 0.04 * rap_signal) * bias.get("confidence", 1.0)

    total = sum(adjusted.values()) or 1.0
    return sorted(((k, v / total) for k, v in adjusted.items()), key=lambda x: x[1], reverse=True)
