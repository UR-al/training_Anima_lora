# -*- coding: utf-8 -*-
"""Tests for near-duplicate image grouping (torch-free, PySide6-free)."""

from gui.native import image_dupes as d


def test_hamming_and_similarity():
    assert d.hamming(0b1011, 0b1001) == 1
    assert d.similarity(5, 5) == 1.0
    assert d.similarity(0, 0b11, bits=64) == 1 - 2 / 64


def test_groups_identical_and_near():
    hashes = {
        "a.png": 0b0000,
        "b.png": 0b0000,  # identical to a
        "c.png": 0b0001,  # 1 bit from a/b
        "z.png": 0b1111_1111,  # far away
    }
    groups = d.group_duplicates(hashes, max_distance=1)
    assert groups == [["a.png", "b.png", "c.png"]]  # z excluded


def test_no_duplicates():
    hashes = {"a": 0, "b": 0xFFFF_FFFF_FFFF_FFFF}
    assert d.group_duplicates(hashes, max_distance=2) == []


def test_threshold_splits_groups():
    hashes = {"a": 0, "b": 0b11, "c": 0b1111}
    # distance a-b=2, b-c=2, a-c=4 → at d=2 a,b,c chain together
    assert d.group_duplicates(hashes, max_distance=2) == [["a", "b", "c"]]
    # at d=1 nothing links
    assert d.group_duplicates(hashes, max_distance=1) == []
