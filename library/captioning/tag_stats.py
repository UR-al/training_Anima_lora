# -*- coding: utf-8 -*-
"""Tag statistics + bulk caption editing — torch/Qt-free, unit-tested.

Powers the Dataset tab's "Tag stats" (dataset-wide tag frequency) and "Bulk tag edit"
(add / remove / replace tags across the selected images' ``.txt`` captions). Tag
tokenization matches the tag sorter (``gui/native/tag_sort.py``): comma-separated,
whitespace-trimmed, blanks dropped. Comparisons are case-insensitive; original casing
and slot order are preserved on write.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path


def split_tags(text: str) -> list[str]:
    """Caption string → tag list (comma-split, trimmed, blanks dropped)."""
    return [t.strip() for t in text.split(",") if t.strip()]


def join_tags(tags: list[str]) -> str:
    """Tag list → caption string (the on-disk ``", "``-joined form)."""
    return ", ".join(tags)


def read_caption(txt_path: str | Path) -> str:
    """Caption text for a ``.txt`` path ('' when missing/unreadable)."""
    p = Path(txt_path)
    try:
        return p.read_text(encoding="utf-8") if p.is_file() else ""
    except OSError:
        return ""


def tag_frequencies(txt_paths: list[str | Path]) -> Counter[str]:
    """Count tag occurrences across caption files (case-insensitive key, one count
    per file even if a tag repeats within a caption). Returns a ``Counter`` keyed on
    the lowercased tag → most_common() gives the dataset ranking."""
    counts: Counter[str] = Counter()
    for p in txt_paths:
        seen = {t.lower() for t in split_tags(read_caption(p))}
        counts.update(seen)
    return counts


def caption_paths_for(dir_path: str | Path, image_names: list[str]) -> list[Path]:
    """``.txt`` sidecar path for each image name under ``dir_path``."""
    d = Path(dir_path)
    return [(d / name).with_suffix(".txt") for name in image_names]


def _apply_one(
    tags: list[str],
    *,
    add: list[str],
    remove: set[str],
    replace: dict[str, str],
) -> list[str]:
    """remove → replace → add, preserving order and de-duplicating (case-insensitive).
    ``remove``/``replace`` keys are lowercased; new tags keep their given casing."""
    out: list[str] = []
    seen: set[str] = set()

    def _push(tag: str) -> None:
        low = tag.lower()
        if low and low not in seen:
            seen.add(low)
            out.append(tag)

    for t in tags:
        low = t.lower()
        if low in remove:
            continue
        _push(replace.get(low, t))
    for t in add:
        _push(t)
    return out


def bulk_edit_captions(
    txt_paths: list[str | Path],
    *,
    add: list[str] | None = None,
    remove: list[str] | None = None,
    replace: dict[str, str] | None = None,
) -> dict:
    """Apply add/remove/replace to each caption file, writing back only when the
    content changes. Missing files are skipped (can't add tags to a caption that
    doesn't exist). Returns ``{"changed", "scanned", "skipped"}``."""
    add = list(add or [])
    remove_set = {t.strip().lower() for t in (remove or []) if t.strip()}
    replace_map = {
        k.strip().lower(): v.strip()
        for k, v in (replace or {}).items()
        if k.strip() and v.strip()
    }
    if not (add or remove_set or replace_map):
        return {"changed": 0, "scanned": 0, "skipped": len(txt_paths)}

    changed = scanned = skipped = 0
    for raw in txt_paths:
        p = Path(raw)
        if not p.is_file():
            skipped += 1
            continue
        scanned += 1
        original = read_caption(p)
        new_tags = _apply_one(
            split_tags(original), add=add, remove=remove_set, replace=replace_map
        )
        new_text = join_tags(new_tags)
        if new_text != original.strip():
            p.write_text(new_text + "\n", encoding="utf-8")
            changed += 1
    return {"changed": changed, "scanned": scanned, "skipped": skipped}


def format_stats(counts: Counter[str], top: int = 0) -> str:
    """Human-readable ``count\\ttag`` table (descending). ``top`` limits rows (0 = all).
    Header line carries the unique-tag total."""
    rows = counts.most_common(top or None)
    lines = [f"{len(counts)} unique tags across captions", ""]
    width = len(str(rows[0][1])) if rows else 1
    lines += [f"{c:>{width}}  {tag}" for tag, c in rows]
    return "\n".join(lines)
