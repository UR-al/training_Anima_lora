# -*- coding: utf-8 -*-
"""Tests for the AI watch-party pure helpers (no SDK / no network)."""

from tools import ai_watch_party as W


def test_metrics_summary():
    state = {
        "step": 50,
        "total_steps": 100,
        "losses": [{"step": i, "loss": 1.0 - i * 0.01} for i in range(1, 51)],
        "config": {"run": "my_lora"},
    }
    s = W.metrics_summary(state)
    assert "run=my_lora" in s and "step=50/100" in s and "best=" in s


def test_to_messages_alternates_and_starts_user():
    tr = [("gpt", "hi"), ("claude", "yo")]
    msgs = W.to_messages(tr, me="claude", context="CTX")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "CTX"


def test_to_messages_empty_transcript():
    msgs = W.to_messages([], me="claude", context="CTX")
    assert msgs == [{"role": "user", "content": "CTX"}]


def test_to_messages_merges_consecutive_users():
    # last speaker is the other AI → merges with the trailing context user turn
    msgs = W.to_messages([("gpt", "a")], me="claude", context="CTX")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert "[gpt] a" in msgs[0]["content"] and "CTX" in msgs[0]["content"]


def test_to_messages_prepends_user_when_starting_assistant():
    # my own line first → would start 'assistant'; a user stub is prepended
    msgs = W.to_messages([("claude", "x")], me="claude", context="CTX")
    assert msgs[0]["role"] == "user"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
