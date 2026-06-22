"""NLTK data bootstrap for the ArtML pipelines."""

import os
import re

import nltk
from nltk.corpus import stopwords as _stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_nltk_ready = False
_nltk_available = False
_LEMMATIZER = WordNetLemmatizer()
_REGEX_TOKEN = re.compile(r"[a-z']+")


def ensure_nltk() -> None:
    global _nltk_ready, _nltk_available
    if _nltk_ready:
        return
    nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)
    for pkg in ("punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger"):
        try:
            nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)
        except Exception:
            pass
    try:
        word_tokenize("test sentence")
        _nltk_available = True
    except LookupError:
        _nltk_available = False
    _nltk_ready = True


def get_lemmatizer() -> WordNetLemmatizer:
    ensure_nltk()
    return _LEMMATIZER


def get_stopwords() -> list[str]:
    ensure_nltk()
    if _nltk_available:
        try:
            return _stopwords.words("english")
        except LookupError:
            pass
    return _FALLBACK_STOPWORDS


def tokenize(text: str) -> list[str]:
    ensure_nltk()
    if _nltk_available:
        try:
            return word_tokenize(text)
        except LookupError:
            pass
    return _REGEX_TOKEN.findall(text.lower())


_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "am", "so", "if", "as", "up", "out", "about", "into", "over", "after", "before",
    "between", "through", "during", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "than", "too", "very", "can",
    "just", "don", "now", "all", "when", "where", "how", "what", "which", "who", "whom",
    "why", "while", "again", "further", "then", "once", "here", "there", "both", "any",
    "both", "off", "own", "same", "until", "because", "while", "of", "s", "t", "ve",
    "ll", "re", "d", "m", "o",
}

