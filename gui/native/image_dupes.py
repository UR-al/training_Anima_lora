# -*- coding: utf-8 -*-
"""Near-duplicate image grouping by perceptual-hash similarity.

Pure Python (no torch, no PySide6) so the grouping is unit-testable. The hashes
themselves are computed by the caller (the Dataset view uses a QImage dHash); here
we only compare 64-bit hashes by Hamming distance and union near-duplicates into
groups. Similarity score = ``1 - hamming / bits`` (1.0 = identical).
"""

from __future__ import annotations


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def similarity(a: int, b: int, bits: int = 64) -> float:
    return 1.0 - hamming(a, b) / bits


def group_duplicates(hashes: dict[str, int], max_distance: int = 10) -> list[list[str]]:
    """Union names whose hashes are within ``max_distance`` bits. Returns the
    multi-member groups, largest first, names sorted within each."""
    names = list(hashes)
    parent = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(names)):
        hi = hashes[names[i]]
        for j in range(i + 1, len(names)):
            if hamming(hi, hashes[names[j]]) <= max_distance:
                union(names[i], names[j])

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted(
        (sorted(g) for g in groups.values() if len(g) > 1),
        key=lambda g: -len(g),
    )
