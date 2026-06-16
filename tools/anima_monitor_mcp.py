#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Anima training-monitor MCP server (stdio).

Exposes the live training monitor to an AI agent (Claude Desktop / Code) so it can
watch the run alongside you — read status / loss / LR, *see* the latest sample
images, tail the captured terminal log, list saved runs, and post analysis notes
back into the dashboard's "AI Analysis" panel.

Reads the same data the web dashboard does (no torch): the training process writes
``library/monitoring/monitor_data/state.json``; samples / logs / run snapshots live
under ``$ANIMA_OUTPUT_DIR`` (default ``output``).

Run (the MCP client usually spawns this): ``python tools/anima_monitor_mcp.py``.
Needs the optional ``mcp`` extra: ``uv sync --extra mcp``.

Claude Desktop config (claude_desktop_config.json):

    "mcpServers": {
      "anima-monitor": {
        "command": "uv",
        "args": ["run", "--extra", "mcp", "python", "tools/anima_monitor_mcp.py"],
        "cwd": "/path/to/training_Anima_lora"
      }
    }
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Bootstrap: import the in-repo data layer (library is normally installed editable,
# but make a direct `python tools/...` invocation work too).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from mcp.server.fastmcp import FastMCP, Image
except ModuleNotFoundError:
    sys.stderr.write(
        "anima_monitor_mcp requires the 'mcp' extra:\n  uv sync --extra mcp\n"
    )
    raise SystemExit(1) from None

from library.monitoring import mcp_data as D  # noqa: E402  (after sys.path bootstrap)

_OUTPUT_DIR = os.environ.get("ANIMA_OUTPUT_DIR") or str(_ROOT / "output")

mcp = FastMCP("anima-monitor")


@mcp.tool()
def training_status() -> dict:
    """Current training status: run name, epoch/step, progress %, latest loss,
    it/s, elapsed/ETA, and the run config. Reads the live monitor state."""
    return D.status(D.read_state())


@mcp.tool()
def loss_curve(last_n: int = 200, ema_alpha: float = 0.05) -> dict:
    """The recent training loss curve (raw + EMA), the validation points, and the
    best (min) loss / val so far. ``last_n`` recent points; ``ema_alpha`` smoothing."""
    return D.loss_curve(D.read_state(), last_n=last_n, ema_alpha=ema_alpha)


@mcp.tool()
def lr_curve(last_n: int = 200) -> dict:
    """The recent learning-rate schedule (last_n points)."""
    return D.lr_curve(D.read_state(), last_n=last_n)


@mcp.tool()
def latest_samples(n: int = 4) -> list[Image]:
    """The n newest generated sample images so you can *look* at them and judge
    quality / artifacts / convergence. Pair with training_status() for the step."""
    out: list[Image] = []
    for _step, path in D.latest_sample_paths(_OUTPUT_DIR, D.read_state(), n=n):
        try:
            out.append(Image(path=str(path)))
        except Exception:
            pass
    return out


@mcp.tool()
def tail_log(n: int = 100) -> str:
    """Last n lines of the captured training terminal log (the newest
    ``$ANIMA_OUTPUT_DIR/logs/*.log``). Empty when training streams only to a TTY."""
    lines = D.tail_log(_OUTPUT_DIR, n=n)
    return "\n".join(lines) if lines else "(no captured log file found)"


@mcp.tool()
def list_runs() -> list[dict]:
    """Saved run snapshots (newest first) with run name / saved time / step /
    final loss — for comparing finished runs."""
    return D.list_runs(_OUTPUT_DIR)


@mcp.tool()
def post_analysis(text: str) -> str:
    """Post an analysis note that appears live in the dashboard's "AI Analysis"
    panel (the share-with-the-user channel). Use for observations, warnings, or
    suggestions (e.g. "val loss diverging since step 800 — consider lowering LR")."""
    note = D.add_note(text, author="ai")
    return f"posted at {note['ts']}"


@mcp.tool()
def set_lr_scale(scale: float) -> dict:
    """Steer the LR live: multiply the *scheduled* LR by ``scale`` (1.0 = normal,
    0.5 = half, 2.0 = double). Takes effect within ~10 steps (training must run
    with --monitor). Clears any active decay. Returns the new control state."""
    return D.set_lr_scale(scale)


@mcp.tool()
def start_lr_decay(k_steps: int, floor: float = 0.0) -> dict:
    """Begin an on-demand cosine decay of the LR from the current scale → ``floor``
    over ``k_steps``, starting now (the "constantcosine, but I pick the moment"
    move). Needs --monitor. Returns the new control state."""
    step = int(D.read_state().get("step") or 0)
    return D.start_lr_decay(step, k_steps, floor)


@mcp.tool()
def reset_lr() -> dict:
    """Back to the scheduled LR (scale 1.0, no decay)."""
    return D.reset_control()


if __name__ == "__main__":
    mcp.run()  # stdio transport
