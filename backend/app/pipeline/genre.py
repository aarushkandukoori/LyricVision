"""Genre classification from lyric lexicon."""

from collections import Counter

GENRE_LEXICON = {
    "rap": {
        "crown", "throne", "chain", "gold", "rings", "lambo", "beat", "drop", "flow",
        "city", "skyline", "king", "queen", "hustle", "grind", "haters", "stack",
        "climb", "rise", "goat", "top", "bottom", "dreams", "hunger", "choir",
        "roll", "streets", "rearview", "crew", "block", "real", "fake",
    },
    "sad": {
        "shadow", "shadows", "rain", "tears", "tear", "cry", "cried", "lost",
        "memories", "memory", "dark", "darkness", "fog", "ghost", "ghosts",
        "broken", "alone", "silence", "falling", "fall", "empty", "gone",
        "wound", "flicker", "flickering", "drowning", "longing", "ache",
    },
    "pop": {
        "dancing", "dance", "neon", "bright", "shine", "glitter", "confetti",
        "summer", "golden", "free", "love", "forever", "young", "colors",
        "rhythm", "alive", "thrive", "sky", "sun", "music", "party",
    },
    "rock": {
        "fire", "storm", "thunder", "lightning", "fight", "battle", "scream",
        "rebel", "wild", "freedom", "highway", "engine", "leather", "blood",
    },
    "electronic": {
        "pulse", "pulses", "synth", "synthesized", "wave", "waves", "circuit",
        "digital", "machine", "glow", "glowing", "underground", "cyber",
        "electric", "frequency", "static",
    },
    "folk": {
        "gravel", "mountain", "mountains", "river", "forest", "whisper",
        "whispers", "barefoot", "wind", "pine", "lantern", "cabin", "wooden",
        "stories", "soil", "harvest",
    },
}


def classify_genre(lyrics: str) -> dict[str, float]:
    import re

    text = lyrics.lower()
    tokens = re.findall(r"[a-z']+", text)
    counts = Counter(tokens)
    scores = {g: 0.0 for g in GENRE_LEXICON}
    for g, vocab in GENRE_LEXICON.items():
        for w in vocab:
            if w in counts:
                scores[g] += counts[w]
    total = sum(scores.values())
    if total == 0:
        return {g: 1.0 / len(scores) for g in scores}
    return {g: round(s / total, 3) for g, s in scores.items()}


def primary_genre(lyrics: str) -> tuple[str, float]:
    scores = classify_genre(lyrics)
    genre = max(scores, key=scores.get)
    return genre, scores[genre]
