"""Lyrics preprocessing — original (V1–V4) and enhanced (V5)."""

import re
import string

from .nlp_setup import get_lemmatizer, get_stopwords, tokenize

EMOTION_PRESERVE = {
    "hurt", "fear", "love", "hate", "sad", "happy", "cry", "pain", "joy",
    "lost", "dark", "light", "night", "dream", "shadow", "fall", "rise",
    "hope", "broken", "free", "alone", "gold", "shine", "fire", "rain",
    "storm", "tears", "ghost", "fog", "glow", "crown", "king", "queen",
    "wound", "silence", "forever", "young", "bright", "wild", "empty",
    "drowning", "flickering", "climbing", "standing", "flying",
}

CUSTOM_STOPWORDS = None
_SECTION_RE = re.compile(r"\[([^\]]+)\]", re.IGNORECASE)


def _custom_stopwords():
    global CUSTOM_STOPWORDS
    if CUSTOM_STOPWORDS is None:
        CUSTOM_STOPWORDS = set(get_stopwords()) - EMOTION_PRESERVE
    return CUSTOM_STOPWORDS


def _apply_regex(text: str) -> str:
    s = text.lower()
    for pat, rep in [
        (r"\[.*?\]", ""),
        (r"\(.*?\)", ""),
        (r"©|™|℗", ""),
        (r"https?://\S+|www\.\S+", ""),
        (r"^[-–—\s]+|[-–—\s]+$", ""),
        (r"\s{2,}", " "),
    ]:
        s = re.sub(pat, rep, s)
    return s


def preprocess_original(text: str) -> list[str]:
    """V1 baseline preprocessing from main.ipynb."""
    s = _apply_regex(text)
    toks = tokenize(s)
    toks = [t for t in toks if t not in string.punctuation]
    return [w for w in toks if w not in get_stopwords()]


def preprocess_lemmatized(text: str) -> list[str]:
    """Improved preprocessing used by V1–V4 in improved_main.ipynb."""
    s = _apply_regex(text)
    toks = tokenize(s)
    toks = [t for t in toks if t not in string.punctuation]
    try:
        lemmatizer = get_lemmatizer()
        return [lemmatizer.lemmatize(w, pos="v") for w in toks if w not in _custom_stopwords()]
    except LookupError:
        return [w for w in toks if w not in _custom_stopwords()]


def preprocess_v5(text: str) -> list[str]:
    """V5 preprocessing with emotion-aware stopwords."""
    s = text.lower()
    for pat, rep in [
        (r"\[.*?\]", ""),
        (r"\(.*?\)", ""),
        (r"©|™|℗", ""),
        (r"https?://\S+|www\.\S+", ""),
        (r"\s{2,}", " "),
    ]:
        s = re.sub(pat, rep, s)
    toks = tokenize(s)
    toks = [t for t in toks if t not in string.punctuation]
    toks = [t for t in toks if t not in _custom_stopwords() and t.isalpha() and len(t) > 1]
    try:
        lemmatizer = get_lemmatizer()
        return [lemmatizer.lemmatize(t, pos="v") for t in toks]
    except LookupError:
        return toks


def split_sections(lyrics: str) -> list[tuple[str, str]]:
    matches = list(_SECTION_RE.finditer(lyrics))
    if not matches:
        return [("verse", lyrics.strip())]
    pieces: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        pre = lyrics[: matches[0].start()].strip()
        if pre:
            pieces.append(("intro", pre))
    for i, m in enumerate(matches):
        label = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(lyrics)
        body = lyrics[start:end].strip()
        if body:
            pieces.append((label, body))
    return pieces
