# -*- coding: utf-8 -*-
"""Caption tag sorter — reorder Anima-format booru tags into the canonical slot
order. Pure Python (no torch, no PySide6) so it is unit-testable and reusable.

Canonical order (matches Anima training captions / the dataset-tagger output)::

    [year, period, quality, meta, safety] [count] [character] [series] [artist] [general]

Within a section order is arbitrary; we keep the input order inside each bucket.

quality / period / year / safety / count are recognized by fixed lists + regex
(they're booru *metadata*, not vision-tagger output). character / series / meta /
artist classification reuses the anima-tagger ``vocab.json`` (a static JSON of
``{name, category}`` — no model load needed); when the vocab is absent those tags
fall back to the hardcoded meta set or to ``general``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Default location of the anima-tagger vocab (download-models target).
DEFAULT_VOCAB_REL = "models/captioners/anima-tagger-v1/vocab.json"

QUALITY_HUMAN = [
    "masterpiece",
    "best quality",
    "good quality",
    "normal quality",
    "low quality",
    "worst quality",
]
_QUALITY_SET = set(QUALITY_HUMAN)
_SCORE_RE = re.compile(r"^score_\d$")  # PonyV7 aesthetic: score_9 … score_1
PERIOD = {"newest", "recent", "mid", "early", "old"}
_YEAR_RE = re.compile(r"^year \d{4}$")
SAFETY = {"safe", "sensitive", "nsfw", "explicit", "questionable"}
# People-count tags (1girl / 2girls / 1boy / 1other / 6+girls / solo / …).
_COUNT_RE = re.compile(
    r"^(solo|no humans|multiple (girls|boys|others)|\d+\+?(girl|boy|other)s?)$"
)
# Common meta tags so they sort right even without the vocab.
META = {
    "highres",
    "absurdres",
    "lowres",
    "anime screenshot",
    "jpeg artifacts",
    "official art",
    "artist name",
    "signature",
    "watermark",
    "commentary",
    "dated",
    "english commentary",
    "web address",
    "twitter username",
}

# Emission order of the buckets (matches the canonical caption example).
_ORDER = [
    "year",
    "period",
    "quality",
    "meta",
    "safety",
    "count",
    "character",
    "series",
    "artist",
    "general",
]


def _norm(tag: str) -> str:
    return tag.strip().lower()


def load_vocab_categories(
    vocab_path: str | Path | None = None,
) -> dict[str, str] | None:
    """Load ``vocab.json`` → ``{tag_name_lower: category}``. Returns None if the
    file is missing/unreadable (caller falls back to rule-based classification)."""
    p = Path(vocab_path) if vocab_path else None
    if not p or not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entries = data.get("tags") if isinstance(data, dict) else data
    out: dict[str, str] = {}
    for e in entries or []:
        if isinstance(e, dict) and e.get("name"):
            out[_norm(str(e["name"]))] = str(e.get("category") or "general")
    return out or None


def classify(tag: str, vocab: dict[str, str] | None = None) -> str:
    """Return the bucket name for a single tag."""
    if tag.strip().startswith("@"):
        return "artist"
    low = _norm(tag)
    if low in _QUALITY_SET or _SCORE_RE.match(low):
        return "quality"
    if low in PERIOD:
        return "period"
    if _YEAR_RE.match(low):
        return "year"
    if low in SAFETY:
        return "safety"
    if _COUNT_RE.match(low):
        return "count"
    if low in META:
        return "meta"
    if vocab:
        cat = vocab.get(low)
        if cat == "metadata":
            return "meta"
        if cat == "character":
            return "character"
        if cat == "copyright":
            return "series"
        if cat == "artist":
            return "artist"
    return "general"


def sort_tags(tags: list[str], vocab: dict[str, str] | None = None) -> list[str]:
    buckets: dict[str, list[str]] = {k: [] for k in _ORDER}
    for t in tags:
        buckets[classify(t, vocab)].append(t)
    out: list[str] = []
    for k in _ORDER:
        out.extend(buckets[k])
    return out


def sort_caption(text: str, vocab: dict[str, str] | None = None) -> str:
    """Split a comma-separated caption, reorder, and re-join."""
    tags = [t.strip() for t in text.split(",") if t.strip()]
    return ", ".join(sort_tags(tags, vocab))
