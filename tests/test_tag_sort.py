# -*- coding: utf-8 -*-
"""Tests for the native GUI caption tag sorter (torch-free, PySide6-free)."""

from gui.native import tag_sort


def test_classify_rule_based():
    assert tag_sort.classify("@nnn yryr") == "artist"
    assert tag_sort.classify("masterpiece") == "quality"
    assert tag_sort.classify("score_5") == "quality"
    assert tag_sort.classify("newest") == "period"
    assert tag_sort.classify("year 2025") == "year"
    assert tag_sort.classify("safe") == "safety"
    assert tag_sort.classify("1girl") == "count"
    assert tag_sort.classify("highres") == "meta"
    assert tag_sort.classify("smile") == "general"


def test_sort_orders_leading_sections():
    cap = "smile, 1girl, @artist x, highres, masterpiece, safe, newest, year 2025"
    out = tag_sort.sort_caption(cap)
    tags = [t.strip() for t in out.split(",")]
    # year → period → quality → meta → safety → count → artist → general
    assert tags == [
        "year 2025",
        "newest",
        "masterpiece",
        "highres",
        "safe",
        "1girl",
        "@artist x",
        "smile",
    ]


def test_vocab_splits_character_and_series():
    vocab = {
        "oomuro sakurako": "character",
        "yuru yuri": "copyright",
        "brown hair": "general",
    }
    cap = "brown hair, yuru yuri, oomuro sakurako, 1girl"
    out = tag_sort.sort_caption(cap, vocab)
    tags = [t.strip() for t in out.split(",")]
    # count → character → series → general
    assert tags == ["1girl", "oomuro sakurako", "yuru yuri", "brown hair"]


def test_insert_keep_tokens_separator():
    vocab = {
        "oomuro sakurako": "character",
        "yuru yuri": "copyright",
        "brown hair": "general",
        "smile": "general",
    }
    cap = "brown hair, yuru yuri, oomuro sakurako, 1girl, @nnn yryr, smile, year 2025, safe"
    out = tag_sort.sort_caption(cap, vocab, insert_sep=True)
    head, sep, gen = out.partition(f" {tag_sort.KEEP_TOKENS_SEPARATOR} ")
    assert sep  # separator present
    assert head == "year 2025, safe, 1girl, oomuro sakurako, yuru yuri, @nnn yryr"
    assert gen == "brown hair, smile"
    # no separator when there is no head or no general bucket
    assert tag_sort.KEEP_TOKENS_SEPARATOR not in tag_sort.sort_caption(
        "smile, brown hair", vocab, insert_sep=True
    )
    assert tag_sort.KEEP_TOKENS_SEPARATOR not in tag_sort.sort_caption(
        "1girl, @x, safe", vocab, insert_sep=True
    )


def test_name_lists_override_vocab_and_sort():
    chars = {"shimarin"}
    series = {"yuru camp"}
    # vocab wrongly calls shimarin general; the user character list wins.
    vocab = {"shimarin": "general", "smile": "general"}
    cap = "smile, yuru camp, shimarin, 1girl, @kagamihara"
    out = tag_sort.sort_caption(cap, vocab, characters=chars, series=series)
    tags = [t.strip() for t in out.split(",")]
    # count → character → series → artist → general
    assert tags == ["1girl", "shimarin", "yuru camp", "@kagamihara", "smile"]


def test_classify_name_lists_underscore_insensitive():
    chars = {"hatsune miku"}
    assert tag_sort.classify("hatsune_miku", characters=chars) == "character"
    assert tag_sort.classify("Hatsune Miku", characters=chars) == "character"


def test_hard_metadata_beats_name_list():
    # a name list can't hijack a hard booru-metadata tag
    assert tag_sort.classify("1girl", characters={"1girl"}) == "count"
    assert tag_sort.classify("safe", series={"safe"}) == "safety"


def test_load_name_set(tmp_path):
    p = tmp_path / "characters.txt"
    p.write_text(
        "# a comment\nHatsune_Miku\n\n  Oomuro Sakurako  # trailing\n",
        encoding="utf-8",
    )
    assert tag_sort.load_name_set(p) == {"hatsune miku", "oomuro sakurako"}
    assert tag_sort.load_name_set(tmp_path / "missing.txt") == set()


def test_missing_vocab_returns_none(tmp_path):
    assert tag_sort.load_vocab_categories(tmp_path / "nope.json") is None


def test_load_vocab_list_and_dict(tmp_path):
    import json

    p = tmp_path / "vocab.json"
    p.write_text(
        json.dumps(
            [
                {"name": "Oomuro Sakurako", "category": "character"},
                {"name": "yuru yuri", "category": "copyright"},
            ]
        ),
        encoding="utf-8",
    )
    vocab = tag_sort.load_vocab_categories(p)
    assert vocab == {"oomuro sakurako": "character", "yuru yuri": "copyright"}
