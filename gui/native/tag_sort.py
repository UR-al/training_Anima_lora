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

# User-owned name lists (one tag per line, '#' comments). These are AUTHORITATIVE for
# character / series classification — a caption tag matching an entry is forced into
# that bucket, overriding the vocab. Swap these two files (no code change) to teach the
# sorter your own characters / works; the head order stays
# metadata → count → character → series → @artist.
DEFAULT_CHARACTERS_REL = "dataset_tags/characters.txt"
DEFAULT_SERIES_REL = "dataset_tags/series.txt"

# Separator inserted right after the @artist tags (everything before it — year …
# artist — is "kept"). Pair with the subset's --keep_tokens_separator so kohya
# keeps exactly the non-general head, per image, regardless of its length.
KEEP_TOKENS_SEPARATOR = "|||"

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


def load_name_set(path: str | Path | None) -> set[str]:
    """Load a one-name-per-line list (``#`` comments, blank lines ignored) into a set
    of normalized names (lowercase, underscores → spaces) for character/series
    matching. Missing/unreadable file → empty set (treated as "no list")."""
    p = Path(path) if path else None
    if not p or not p.exists():
        return set()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return set()
    out: set[str] = set()
    for line in text.splitlines():
        name = line.split("#", 1)[0].strip()
        if name:
            out.add(_norm(name).replace("_", " "))
    return out


def classify(
    tag: str,
    vocab: dict[str, str] | None = None,
    characters: set[str] | None = None,
    series: set[str] | None = None,
) -> str:
    """Return the bucket name for a single tag.

    ``characters`` / ``series`` (user name lists) are authoritative — they override the
    vocab — but yield to the hard booru-metadata rules (a real name is never ``1girl``
    or ``safe``), so they slot in just before the META set / vocab lookup."""
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
    key = low.replace("_", " ")  # match the load_name_set normalization
    if characters and key in characters:
        return "character"
    if series and key in series:
        return "series"
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


def _split_sorted(
    tags: list[str],
    vocab: dict[str, str] | None,
    characters: set[str] | None = None,
    series: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Reorder and return (head, general) where head = year…artist (the "kept"
    tags) and general = everything else, both in canonical order."""
    buckets: dict[str, list[str]] = {k: [] for k in _ORDER}
    for t in tags:
        buckets[classify(t, vocab, characters, series)].append(t)
    head: list[str] = []
    for k in _ORDER:
        if k != "general":
            head.extend(buckets[k])
    return head, buckets["general"]


def sort_tags(
    tags: list[str],
    vocab: dict[str, str] | None = None,
    characters: set[str] | None = None,
    series: set[str] | None = None,
) -> list[str]:
    head, general = _split_sorted(tags, vocab, characters, series)
    return head + general


def sort_caption(
    text: str,
    vocab: dict[str, str] | None = None,
    insert_sep: bool = False,
    characters: set[str] | None = None,
    series: set[str] | None = None,
) -> str:
    """Split a comma-separated caption, reorder, and re-join. With ``insert_sep``,
    place ``KEEP_TOKENS_SEPARATOR`` between the kept head (year…@artist) and the
    general tags so a per-image keep_tokens boundary is encoded in the caption."""
    tags = [t.strip() for t in text.split(",") if t.strip()]
    head, general = _split_sorted(tags, vocab, characters, series)
    if insert_sep and head and general:
        return ", ".join(head) + f" {KEEP_TOKENS_SEPARATOR} " + ", ".join(general)
    return ", ".join(head + general)
