# -*- coding: utf-8 -*-
"""Tests for the torch-free parts of the Qwen captioner (config + prompt + finalize).
The model load / generation is heavy + model-specific and not unit-tested here."""

import pytest

from library.captioning import qwen_caption as qc


def _write_cfg(tmp_path):
    p = tmp_path / "qwen_caption.toml"
    p.write_text(
        'model_path = "m"\nloader = "qwen2_5_vl"\ntrigger = "mychar"\n'
        '[prompts]\ntags = "as tags {trigger}"\nnatural = "a sentence"\n',
        encoding="utf-8",
    )
    return p


def test_load_caption_config(tmp_path):
    cfg = qc.load_caption_config(_write_cfg(tmp_path))
    assert cfg["model_path"] == "m" and cfg["loader"] == "qwen2_5_vl"
    assert cfg["prompts"]["natural"] == "a sentence"


def test_load_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        qc.load_caption_config(tmp_path / "nope.toml")


def test_build_prompt_modes_and_trigger(tmp_path):
    cfg = qc.load_caption_config(_write_cfg(tmp_path))
    assert qc.build_prompt(cfg, "tags") == "as tags mychar"  # {trigger} filled
    assert qc.build_prompt(cfg, "natural") == "a sentence"
    with pytest.raises(ValueError):
        qc.build_prompt(cfg, "bogus")


def test_finalize_prepends_trigger_and_tidies():
    assert qc._finalize("  a,  b ,", "char") == "char, a, b"
    # trigger already present → not duplicated
    assert qc._finalize("char, a", "char") == "char, a"
    assert qc._finalize("a, b", "") == "a, b"  # no trigger


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
