"""Keyword extraction for V1–V4 (TF-IDF + frequency) and V5 (visual metaphors)."""

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .nlp_setup import tokenize
from .preprocess import EMOTION_PRESERVE, preprocess_lemmatized, preprocess_v5

IMAGEABLE_NOUNS = {
    "rain", "fog", "storm", "clouds", "sky", "stars", "moon", "sun", "sunset", "sunrise",
    "lightning", "thunder", "snow", "mist", "dawn", "dusk", "horizon",
    "city", "skyline", "street", "streets", "alley", "rooftop", "window", "windows",
    "subway", "bridge", "park", "bench", "sidewalk", "neon", "sign", "lamppost",
    "streetlight", "lamp", "lantern",
    "umbrella", "chain", "crown", "ring", "rings", "car", "lambo", "mirror", "rearview",
    "phone", "letter", "letters", "photograph", "candle", "match", "fire", "smoke",
    "flame", "glass", "vinyl", "record", "tape",
    "forest", "river", "ocean", "sea", "wave", "waves", "mountain", "mountains", "tree",
    "trees", "flower", "flowers", "field", "fields", "road", "path", "trail", "grass",
    "figure", "silhouette", "hands", "face", "hair", "eyes", "tear", "tears", "shoulders",
    "back", "feet",
    "light", "lights", "glow", "shadow", "shadows", "reflection", "silver", "gold",
    "glitter", "confetti", "sparks", "dust",
    "wall", "door", "floor", "room", "ceiling", "corridor",
}

VISUAL_ADJECTIVES = {
    "bright", "dark", "soft", "warm", "cold", "glowing", "flickering", "golden", "silver",
    "neon", "foggy", "rainy", "empty", "lonely", "dusty", "broken", "shattered", "open",
    "closed", "wide", "narrow", "distant", "blurry", "sharp", "wet", "dry", "quiet", "loud",
    "crimson", "blue", "red", "green", "yellow", "black", "white", "pink",
    "purple", "orange", "gold", "silver", "indigo", "teal", "magenta",
}

GENRE_ANCHOR_WORDS = {
    "sad": {"rain", "shadow", "tears", "memory", "fog", "ghost", "silence", "alone", "broken", "flickering", "candle", "window"},
    "pop": {"glitter", "neon", "confetti", "golden", "sun", "dance", "party", "bright", "sky", "colors", "sparkle", "shine"},
    "rap": {"crown", "gold", "chain", "skyline", "throne", "city", "king", "rings", "lights", "stack", "grind", "rise"},
    "rock": {"fire", "storm", "thunder", "engine", "leather", "highway", "blood", "rebel"},
    "electronic": {"neon", "pulse", "synth", "wave", "circuit", "glow", "machine", "underground"},
    "folk": {"mountain", "river", "wood", "lantern", "field", "wind", "barefoot", "cabin"},
}

EMOTIONAL_WORDS = {
    "hurt", "fear", "love", "hate", "sad", "happy", "cry", "pain", "joy", "lost",
    "dark", "light", "night", "dream",
}


def extract_keywords_tfidf(text: str, max_features: int = 15) -> list[str]:
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    vectorizer.fit([text])
    return vectorizer.get_feature_names_out().tolist()[:10]


def extract_keywords_frequency(tokens: list[str]) -> list[str]:
    word_freq = Counter(tokens)
    for token in tokens:
        if token in EMOTIONAL_WORDS:
            word_freq[token] *= 2
    return [word for word, _ in word_freq.most_common(10)]


def extract_keywords_v1_v4(lyrics: str) -> list[str]:
    tokens = preprocess_lemmatized(lyrics)
    tfidf_kw = extract_keywords_tfidf(lyrics)
    freq_kw = extract_keywords_frequency(tokens)
    return list(dict.fromkeys(tfidf_kw + freq_kw))[:7]


def extract_visual_metaphors(lyrics: str, top_k: int = 6) -> list[str]:
    text = re.sub(r"\[.*?\]", "", lyrics).lower()
    tokens = tokenize(text)
    try:
        from nltk import pos_tag
        tagged = pos_tag(tokens)
    except LookupError:
        tagged = [(t, "NN") for t in tokens]
    candidates: Counter[str] = Counter()

    for i in range(len(tagged) - 1):
        w1, t1 = tagged[i]
        w2, t2 = tagged[i + 1]
        if not (w1.isalpha() and w2.isalpha()):
            continue
        if w2 not in IMAGEABLE_NOUNS:
            continue
        if t1.startswith("JJ") and (w1 in VISUAL_ADJECTIVES or t1 == "JJ"):
            candidates[f"{w1} {w2}"] += 2
        elif t1.startswith("NN") and w1 in IMAGEABLE_NOUNS:
            candidates[f"{w1} {w2}"] += 1

    for w, t in tagged:
        if t.startswith("NN") and w in IMAGEABLE_NOUNS:
            candidates[w] += 1

    return [c for c, _ in candidates.most_common(top_k)]


def extract_keywords_v5(lyrics: str, genre: str, top_k: int = 8) -> list[str]:
    tokens = preprocess_v5(lyrics)
    raw_text = " ".join(tokens)

    try:
        vec = TfidfVectorizer(max_features=40, stop_words="english", ngram_range=(1, 1))
        X = vec.fit_transform([raw_text])
        feature_array = np.array(vec.get_feature_names_out())
        tfidf_scores = X.toarray()[0]
        tfidf_top = feature_array[np.argsort(tfidf_scores)[::-1]][:12].tolist()
    except ValueError:
        tfidf_top = []

    freq = Counter(tokens)
    anchors = GENRE_ANCHOR_WORDS.get(genre, set())
    for tok in list(freq.keys()):
        if tok in anchors:
            freq[tok] = int(freq[tok] * 2.5)
        elif tok in EMOTION_PRESERVE:
            freq[tok] = int(freq[tok] * 1.6)
    freq_top = [w for w, _ in freq.most_common(12)]

    visual = extract_visual_metaphors(lyrics, top_k=6)
    merged: list[str] = []
    seen: set[str] = set()
    for src in (visual, freq_top, tfidf_top):
        for w in src:
            if w not in seen:
                merged.append(w)
                seen.add(w)
    return merged[:top_k]
