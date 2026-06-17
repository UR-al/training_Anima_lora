# -*- coding: utf-8 -*-
"""Tests for tag statistics + bulk caption editing (torch/Qt-free)."""

from pathlib import Path

from library.captioning import tag_stats as ts


def _w(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_split_and_join_roundtrip():
    assert ts.split_tags("a ,  b,,c ") == ["a", "b", "c"]
    assert ts.join_tags(["a", "b"]) == "a, b"


def test_tag_frequencies_dedup_per_file(tmp_path):
    _w(tmp_path / "1.txt", "cat, dog, cat")  # cat counted once for this file
    _w(tmp_path / "2.txt", "Cat, bird")  # case-insensitive key
    counts = ts.tag_frequencies([tmp_path / "1.txt", tmp_path / "2.txt"])
    assert counts["cat"] == 2 and counts["dog"] == 1 and counts["bird"] == 1
    assert counts.most_common(1)[0][0] == "cat"


def test_bulk_add_remove_replace(tmp_path):
    f = _w(tmp_path / "a.txt", "1girl, blue hair, smile")
    res = ts.bulk_edit_captions(
        [f],
        add=["masterpiece"],
        remove=["smile"],
        replace={"blue hair": "red hair"},
    )
    assert res == {"changed": 1, "scanned": 1, "skipped": 0}
    assert f.read_text(encoding="utf-8").strip() == "1girl, red hair, masterpiece"


def test_bulk_dedup_and_case_insensitive(tmp_path):
    f = _w(tmp_path / "a.txt", "1girl, Smile")
    ts.bulk_edit_captions([f], add=["smile", "1girl"])  # both already present (ci)
    assert f.read_text(encoding="utf-8").strip() == "1girl, Smile"  # unchanged order


def test_bulk_no_op_when_unchanged(tmp_path):
    f = _w(tmp_path / "a.txt", "a, b")
    res = ts.bulk_edit_captions([f], remove=["zzz"])  # nothing matches
    assert res["changed"] == 0


def test_bulk_skips_missing_file(tmp_path):
    res = ts.bulk_edit_captions([tmp_path / "nope.txt"], add=["x"])
    assert res == {"changed": 0, "scanned": 0, "skipped": 1}


def test_bulk_empty_edit_is_noop(tmp_path):
    f = _w(tmp_path / "a.txt", "a, b")
    res = ts.bulk_edit_captions([f])
    assert res["changed"] == 0 and res["skipped"] == 1


def test_format_stats(tmp_path):
    counts = ts.tag_frequencies([_w(tmp_path / "1.txt", "a, a, b")])
    out = ts.format_stats(counts)
    assert "unique tags" in out and "a" in out and "b" in out


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
