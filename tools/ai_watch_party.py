#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI watch party — two AIs (Claude + GPT) watch a training run together and talk.

Each round, both models are handed the live metrics (status + loss trend) and the
latest sample image (both are vision models), and they take turns discussing
training health and next moves. Every message is posted to the dashboard's "AI
Analysis" panel (via the shared notes file), so you watch the debate live.

This is the orchestration MCP alone doesn't provide — MCP wires one agent to
tools; here a small loop relays two agents and shares the run as their eyes (read
through the torch-free :mod:`library.monitoring.mcp_data`).

Run: ``python tools/ai_watch_party.py [--rounds N] [--interval 30]``. Needs the
optional ``watch`` extra (``uv sync --extra watch``) and ANTHROPIC_API_KEY +
OPENAI_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from library.monitoring import mcp_data as D  # noqa: E402  (after sys.path bootstrap)

_SYSTEM = (
    "You are {me}, an expert diffusion-model LoRA training analyst, watching a LIVE "
    "training run together with another AI ({other}). You both see the same metrics "
    "and the latest sample image. Discuss training health — loss trend / plateau / "
    "divergence, overfitting, sample quality & artifacts — and debate concrete next "
    "actions (adjust LR, keep going, stop, change data). Keep each message to 2–4 "
    "sentences. Address {other} directly, build on or push back on their points. Be "
    "specific and cite the numbers/sample you see."
)


# --------------------------------------------------------------------------- #
# Pure helpers (no SDK) — testable.
# --------------------------------------------------------------------------- #
def metrics_summary(state: dict) -> str:
    s = D.status(state)
    c = D.loss_curve(state, last_n=50)
    pts = c["points"]
    trend = ""
    if len(pts) >= 2:
        d = pts[-1]["ema"] - pts[0]["ema"]
        trend = f", EMA {'↓' if d < 0 else '↑'}{abs(d):.4f} over last {len(pts)}"
    return (
        f"run={s.get('run')} step={s.get('step')}/{s.get('total_steps')} "
        f"({s.get('progress_pct')}%) loss={s.get('latest_loss')} "
        f"best={c.get('best_loss')} best_val={c.get('best_val')} "
        f"it/s={s.get('speed_it_s')} eta_s={s.get('eta_s')}{trend}"
    )


def to_messages(transcript: list[tuple[str, str]], me: str, context: str) -> list[dict]:
    """Transcript (+ trailing fresh-context user turn) → role-tagged messages for
    ``me``: the other speaker's lines are ``user``, mine are ``assistant``.
    Consecutive same-role turns are merged; the list always starts with ``user``."""
    raw: list[dict] = []
    for speaker, text in transcript:
        role = "assistant" if speaker == me else "user"
        raw.append({"role": role, "content": f"[{speaker}] {text}"})
    raw.append({"role": "user", "content": context})
    merged: list[dict] = []
    for m in raw:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "(let's analyze the run)"})
    return merged


def _latest_sample_b64(output_dir, state) -> str | None:
    samples = D.latest_sample_paths(output_dir, state, n=1)
    if not samples:
        return None
    try:
        return base64.b64encode(samples[0][1].read_bytes()).decode("ascii")
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Agents (lazy SDK imports).
# --------------------------------------------------------------------------- #
class ClaudeAgent:
    name = "claude"

    def __init__(self, model: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def respond(self, system, msgs, img_b64):
        out = [dict(m) for m in msgs]
        last = out[-1]
        content = [{"type": "text", "text": last["content"]}]
        if img_b64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                }
            )
        last["content"] = content
        resp = self.client.messages.create(
            model=self.model, max_tokens=400, system=system, messages=out
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class GptAgent:
    name = "gpt"

    def __init__(self, model: str):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def respond(self, system, msgs, img_b64):
        out = [{"role": "system", "content": system}] + [dict(m) for m in msgs]
        last = out[-1]
        content = [{"type": "text", "text": last["content"]}]
        if img_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                }
            )
        last["content"] = content
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=400, messages=out
        )
        return resp.choices[0].message.content or ""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Two-AI training watch party (Claude + GPT)"
    )
    ap.add_argument("--rounds", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument(
        "--interval", type=float, default=30.0, help="seconds between rounds"
    )
    ap.add_argument(
        "--claude-model",
        default=os.environ.get("WATCH_CLAUDE_MODEL", "claude-sonnet-4-6"),
    )
    ap.add_argument("--gpt-model", default=os.environ.get("WATCH_GPT_MODEL", "gpt-4o"))
    ap.add_argument(
        "--output_dir",
        default=os.environ.get("ANIMA_OUTPUT_DIR") or str(_ROOT / "output"),
    )
    ap.add_argument("--no-images", action="store_true", help="don't send sample images")
    args = ap.parse_args()

    try:
        agents = [ClaudeAgent(args.claude_model), GptAgent(args.gpt_model)]
    except ModuleNotFoundError:
        sys.stderr.write("watch party needs the 'watch' extra: uv sync --extra watch\n")
        return 1

    transcript: list[tuple[str, str]] = []
    rounds = 0
    print(
        "[watch] starting — posting to the dashboard AI Analysis panel. Ctrl-C to stop."
    )
    try:
        while args.rounds == 0 or rounds < args.rounds:
            state = D.read_state()
            ctx = "Live metrics: " + metrics_summary(state)
            img = None if args.no_images else _latest_sample_b64(args.output_dir, state)
            for agent in agents:
                other = next(a.name for a in agents if a is not agent)
                system = _SYSTEM.format(me=agent.name, other=other)
                msgs = to_messages(transcript, agent.name, ctx)
                try:
                    reply = agent.respond(system, msgs, img).strip()
                except Exception as exc:  # noqa: BLE001
                    reply = f"(error: {exc})"
                transcript.append((agent.name, reply))
                transcript[:] = transcript[-20:]  # bound context
                D.add_note(reply, author=agent.name)
                print(f"\n[{agent.name}] {reply}")
            rounds += 1
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[watch] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
