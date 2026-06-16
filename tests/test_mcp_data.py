# -*- coding: utf-8 -*-
"""Tests for the monitor MCP data layer (stdlib-only)."""

import json

from library.monitoring import mcp_data as m


def _state():
    return {
        "epoch": 2,
        "step": 50,
        "total_steps": 100,
        "speed": 1.5,
        "start_time": None,
        "losses": [{"step": i, "loss": 1.0 - i * 0.01} for i in range(1, 51)],
        "val_losses": [{"step": 25, "loss": 0.6}, {"step": 50, "loss": 0.5}],
        "lr_history": [{"step": i, "lr": 1e-4} for i in range(1, 51)],
        "samples": [
            {"path": "x_0010.png", "step": 10},
            {"path": "x_0050.png", "step": 50},
        ],
        "config": {"run": "my_lora", "method": "lora"},
    }


def test_status():
    s = m.status(_state())
    assert s["run"] == "my_lora"
    assert s["step"] == 50 and s["total_steps"] == 100
    assert s["progress_pct"] == 50.0
    assert abs(s["latest_loss"] - 0.5) < 1e-9


def test_loss_curve_ema_and_best():
    c = m.loss_curve(_state(), last_n=10, ema_alpha=0.1)
    assert len(c["points"]) == 10
    assert "ema" in c["points"][0]
    assert c["best_loss"] == 0.5  # min over full history
    assert c["best_val"] == 0.5
    assert len(c["val"]) == 2


def test_latest_sample_paths(tmp_path):
    sd = tmp_path / "sample"
    sd.mkdir()
    (sd / "x_0050.png").write_bytes(b"\x89PNG")
    got = m.latest_sample_paths(tmp_path, _state(), n=4)
    # only the existing file (x_0050) returned, newest first
    assert [step for step, _ in got] == [50]


def test_tail_log(tmp_path):
    ld = tmp_path / "logs"
    ld.mkdir()
    (ld / "run.log").write_text(
        "\n".join(f"line {i}" for i in range(20)), encoding="utf-8"
    )
    assert m.tail_log(tmp_path, n=5) == [f"line {i}" for i in range(15, 20)]
    assert m.tail_log(tmp_path / "nope", n=5) == []


def test_list_runs(tmp_path):
    r = tmp_path / "runs" / "my_lora_20260101-000000"
    r.mkdir(parents=True)
    (r / "meta.json").write_text(
        json.dumps({"run": "my_lora", "final_loss": 0.4}), encoding="utf-8"
    )
    runs = m.list_runs(tmp_path)
    assert len(runs) == 1 and runs[0]["id"] == "my_lora_20260101-000000"
    assert runs[0]["final_loss"] == 0.4


def test_notes_roundtrip(tmp_path):
    np = tmp_path / "ai_notes.json"
    assert m.read_notes(np) == []
    note = m.add_note("loss plateaued at step 50", np)
    assert note["text"].startswith("loss plateaued")
    notes = m.read_notes(np)
    assert len(notes) == 1 and notes[0]["author"] == "ai"
