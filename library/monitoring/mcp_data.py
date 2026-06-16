# -*- coding: utf-8 -*-
"""Data layer for the Anima monitor MCP server (and the dashboard AI-notes panel).

Pure stdlib (no torch, no mcp, no PySide6) so it is unit-testable and importable
anywhere. Reads the live monitor state the *training* process writes to
``library/monitoring/monitor_data/state.json`` (the MCP server runs in a separate
process, so it always reads from disk, never in-memory), plus the sample PNGs /
captured logs / saved run snapshots under ``<output_dir>/``. ``add_note`` /
``read_notes`` back the AI-Analysis write-back loop.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

# state.json is written next to this module (train_monitor.MONITOR_DIR).
MONITOR_DIR = Path(__file__).resolve().parent / "monitor_data"
STATE_PATH = MONITOR_DIR / "state.json"
NOTES_PATH = MONITOR_DIR / "ai_notes.json"
CONTROL_PATH = MONITOR_DIR / "control.json"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def read_state(state_path: str | Path | None = None) -> dict:
    p = Path(state_path) if state_path else STATE_PATH
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def status(state: dict) -> dict:
    """Compact training status (live snapshot)."""
    losses = state.get("losses") or []
    step = state.get("step") or 0
    total = state.get("total_steps") or 0
    start = state.get("start_time")
    elapsed = (time.time() - start) if start else 0.0
    eta = ((total - step) * (elapsed / step)) if (step and total and elapsed) else None
    cfg = state.get("config") or {}
    return {
        "run": cfg.get("run"),
        "epoch": state.get("epoch"),
        "step": step,
        "total_steps": total,
        "progress_pct": round(100 * step / total, 1) if total else None,
        "latest_loss": losses[-1].get("loss") if losses else None,
        "speed_it_s": state.get("speed"),
        "elapsed_s": round(elapsed, 1) if elapsed else None,
        "eta_s": round(eta, 1) if eta else None,
        "config": cfg,
    }


def loss_curve(state: dict, last_n: int = 200, ema_alpha: float = 0.05) -> dict:
    losses = (state.get("losses") or [])[-last_n:]
    vals = state.get("val_losses") or []
    ev = _ema([p.get("loss", 0.0) for p in losses], ema_alpha)
    return {
        "points": [
            {"step": p.get("step"), "loss": p.get("loss"), "ema": ev[i]}
            for i, p in enumerate(losses)
        ],
        "val": [{"step": p.get("step"), "loss": p.get("loss")} for p in vals],
        "best_loss": min(
            (p.get("loss", float("inf")) for p in state.get("losses") or []),
            default=None,
        ),
        "best_val": min((p.get("loss", float("inf")) for p in vals), default=None),
    }


def lr_curve(state: dict, last_n: int = 200) -> dict:
    lr = (state.get("lr_history") or [])[-last_n:]
    return {"points": [{"step": p.get("step"), "lr": p.get("lr")} for p in lr]}


def latest_sample_paths(
    output_dir: str | Path, state: dict, n: int = 4
) -> list[tuple[int, Path]]:
    """The n newest sample images (step, path) that exist on disk."""
    sample_dir = Path(output_dir) / "sample"
    out: list[tuple[int, Path]] = []
    for s in reversed(state.get("samples") or []):
        fn = Path(str(s.get("path", ""))).name
        p = sample_dir / fn
        if fn and p.exists():
            out.append((int(s.get("step") or 0), p))
        if len(out) >= n:
            break
    return out


def tail_log(output_dir: str | Path, n: int = 100) -> list[str]:
    """Last n lines of the most-recently-written ``<output_dir>/logs/*.log``."""
    log_dir = Path(output_dir) / "logs"
    if not log_dir.is_dir():
        return []
    logs = [f for f in log_dir.glob("*.log") if f.is_file()]
    if not logs:
        return []
    newest = max(logs, key=lambda f: f.stat().st_mtime)
    try:
        lines = newest.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def list_runs(output_dir: str | Path) -> list[dict]:
    """Saved run-snapshot summaries (newest first)."""
    runs_dir = Path(output_dir) / "runs"
    out: list[dict] = []
    if runs_dir.is_dir():
        for sub in sorted(runs_dir.iterdir(), reverse=True):
            mf = sub / "meta.json"
            if mf.is_file():
                try:
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    m["id"] = sub.name
                    out.append(m)
                except (OSError, ValueError):
                    pass
    return out


def add_note(
    text: str, notes_path: str | Path | None = None, author: str = "ai"
) -> dict:
    """Append an AI-analysis note (shown in the dashboard's AI Analysis panel)."""
    p = Path(notes_path) if notes_path else NOTES_PATH
    notes = read_notes(p)
    note = {"ts": time.strftime("%H:%M:%S"), "author": author, "text": str(text)}
    notes.append(note)
    notes = notes[-100:]  # keep the last 100
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(notes), encoding="utf-8")
    return note


def read_notes(notes_path: str | Path | None = None) -> list[dict]:
    p = Path(notes_path) if notes_path else NOTES_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


# --------------------------------------------------------------------------- #
# Runtime LR control — a tiny control.json the training loop polls (gated on
# --monitor). ``lr_scale`` is a live multiplier on the *scheduled* LR (the
# scheduler resets lr every step, so multiplying never compounds). ``decay`` is an
# on-demand cosine ramp from ``from`` → ``floor`` over ``k_steps`` starting at
# ``start_step`` (the "constantcosine, but I pick the moment" feel).
# --------------------------------------------------------------------------- #
def read_control(control_path: str | Path | None = None) -> dict:
    p = Path(control_path) if control_path else CONTROL_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_control(control: dict, control_path: str | Path | None = None) -> dict:
    p = Path(control_path) if control_path else CONTROL_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(control), encoding="utf-8")
    return control


def set_lr_scale(scale: float, control_path: str | Path | None = None) -> dict:
    """Set the live LR multiplier (1.0 = scheduled, 0.5 = half, 2.0 = double).
    Clears any active decay."""
    ctrl = read_control(control_path)
    ctrl["lr_scale"] = max(0.0, float(scale))
    ctrl["decay"] = None
    return write_control(ctrl, control_path)


def start_lr_decay(
    start_step: int,
    k_steps: int,
    floor: float = 0.0,
    control_path: str | Path | None = None,
) -> dict:
    """Begin an on-demand cosine decay from the current scale → ``floor`` over
    ``k_steps`` starting at ``start_step``."""
    ctrl = read_control(control_path)
    frm = float(ctrl.get("lr_scale", 1.0) or 1.0)
    ctrl["decay"] = {
        "start_step": int(start_step),
        "k_steps": max(1, int(k_steps)),
        "from": frm,
        "floor": max(0.0, float(floor)),
    }
    return write_control(ctrl, control_path)


def reset_control(control_path: str | Path | None = None) -> dict:
    """Back to the scheduled LR (scale 1.0, no decay)."""
    return write_control({"lr_scale": 1.0, "decay": None}, control_path)


def effective_lr_scale(control: dict, step: int) -> float:
    """The multiplier to apply to the scheduled LR at ``step`` given the control."""
    scale = float(control.get("lr_scale", 1.0) or 1.0)
    d = control.get("decay")
    if isinstance(d, dict):
        k = max(1, int(d.get("k_steps", 1)))
        p = (step - int(d.get("start_step", step))) / k
        p = min(1.0, max(0.0, p))
        frm = float(d.get("from", scale))
        floor = float(d.get("floor", 0.0))
        return floor + (frm - floor) * 0.5 * (1 + math.cos(math.pi * p))
    return scale
