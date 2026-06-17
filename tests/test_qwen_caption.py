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


def test_ollama_base_url_default_and_v1_trim():
    assert qc.ollama_base_url({}) == "http://localhost:11434"  # native, no /v1
    assert (
        qc.ollama_base_url({"base_url": "http://x:1/v1"}) == "http://x:1"
    )  # /v1 trimmed
    assert qc.ollama_base_url({"base_url": "http://x:1/"}) == "http://x:1"


def test_ollama_chat_request_shape():
    url, payload = qc.ollama_chat_request(
        "http://h:11434", "qwen-cap", "BASE64", "describe", {"max_new_tokens": 64}
    )
    assert url == "http://h:11434/api/chat"
    assert payload["model"] == "qwen-cap" and payload["stream"] is False
    msg = payload["messages"][0]
    assert msg["content"] == "describe" and msg["images"] == ["BASE64"]
    assert payload["options"]["num_predict"] == 64


def test_resolve_openai_endpoint_blank_falls_to_env():
    base, key = qc.resolve_openai_endpoint({})
    assert base is None and key is None  # → real OpenAI + OPENAI_API_KEY env
    _, key = qc.resolve_openai_endpoint({"api_key": "sk-x"})
    assert key == "sk-x"


def test_ollama_is_openai_compat():
    assert "ollama" in qc.OPENAI_COMPAT_LOADERS


def test_finalize_prepends_trigger_and_tidies():
    assert qc._finalize("  a,  b ,", "char") == "char, a, b"
    # trigger already present → not duplicated
    assert qc._finalize("char, a", "char") == "char, a"
    assert qc._finalize("a, b", "") == "a, b"  # no trigger


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
