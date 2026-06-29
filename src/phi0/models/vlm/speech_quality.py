"""Heuristics for VLM agent speech sanity checks (tests / debug only)."""

from __future__ import annotations

import re
from collections import Counter


def word_repetition_ratio(text: str) -> float:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 6:
        return 1.0
    top = Counter(words).most_common(1)[0][1]
    return float(top) / float(len(words))


def looks_degraded(text: str) -> bool:
    """Repetitive stutter or extreme token collapse (Psi0 HE LM), not normal stopwords."""
    t = text.strip()
    if len(t) < 15:
        return True
    if bool(re.search(r"(\b\w+\b)(?:\s+\1){2,}", t.lower())):
        return True
    words = re.findall(r"[a-zA-Z']+", t.lower())
    if len(words) < 6:
        return True
    top_word, top_count = Counter(words).most_common(1)[0]
    rep = float(top_count) / float(len(words))
    if rep > 0.22:
        return True
    if top_count >= 8 and top_word in {
        "collect",
        "aluminum",
        "coffee",
        "green",
        "pink",
        "soap",
        "duck",
        "square",
    }:
        return True
    return False


def looks_coherent(text: str) -> bool:
    """Fluent enough for official-vs-Psi0 contrast tests."""
    t = text.strip()
    if len(t) < 40:
        return False
    if looks_degraded(t):
        return False
    words = re.findall(r"[a-zA-Z']+", t.lower())
    return len(words) >= 12
