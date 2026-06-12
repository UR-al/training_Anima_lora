# -*- coding: utf-8 -*-
"""Stdlib HTTP control panel: configure -> launch -> monitor.

Serves a single-page form whose dropdowns are populated from the live registries
(methods, presets, the ~89-optimizer zoo, schedulers), builds the exact
``train.py`` command via the shared ``scripts.tasks._common`` helpers, and spawns
training as a detached subprocess. The live loss/LR dashboard is the existing web
monitor (``--monitor``), which this panel links to.

No third-party deps — only the Python stdlib + the trainer it launches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
HTML_FILE = Path(__file__).resolve().parent / "index.html"

# Single in-flight training process (this panel launches one run at a time).
_STATE: dict = {"proc": None, "cmd": None, "monitor_url": None, "started_at": None}


# --------------------------------------------------------------------------- #
# Option registries (drive the form dropdowns)
# --------------------------------------------------------------------------- #
def list_methods() -> list[str]:
    d = ROOT / "configs" / "methods"
    out = sorted(p.stem for p in d.glob("*.toml")) if d.is_dir() else []
    return out or ["lora"]


def list_presets() -> list[str]:
    import tomllib

    p = ROOT / "configs" / "presets.toml"
    try:
        return list(tomllib.loads(p.read_text(encoding="utf-8")).keys()) or ["default"]
    except Exception:
        return ["default"]


def list_optimizers() -> list[str]:
    """kohya built-ins first, then the vendored zoo (class names, available only)."""
    builtins = [
        "AdamW", "AdamW8bit", "PagedAdamW8bit", "Lion", "Prodigy",
        "DAdaptAdam", "Adafactor", "RAdamScheduleFree", "AdamWScheduleFree",
    ]
    custom: list[str] = []
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from LoraEasyCustomOptimizer import OPTIMIZERS  # type: ignore

        custom = sorted({cls.__name__ for cls in OPTIMIZERS.values()})
    except Exception:
        pass
    seen, out = set(), []
    for name in builtins + custom:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def list_schedulers() -> list[str]:
    return [
        "cosine", "cosine_with_restarts", "constant", "constant_with_warmup",
        "linear", "polynomial", "warmup_stable_decay",
        "LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts",
        "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
    ]


def options() -> dict:
    return {
        "methods": list_methods(),
        "presets": list_presets(),
        "optimizers": list_optimizers(),
        "schedulers": list_schedulers(),
    }


# --------------------------------------------------------------------------- #
# Command building + launch
# --------------------------------------------------------------------------- #
def build_command(form: dict) -> list[str]:
    """Translate the form dict into the exact train.py launch command."""
    from scripts.tasks._common import build_launch_cmd, build_method_args

    method = (form.get("method") or "lora").strip()
    preset = (form.get("preset") or "default").strip()
    extra: list[str] = []

    def add(flag: str, key: str) -> None:
        v = form.get(key)
        if v not in (None, "", []):
            extra.extend([flag, str(v)])

    add("--optimizer_type", "optimizer_type")
    add("--learning_rate", "learning_rate")
    add("--dataset_config", "dataset_config")
    add("--max_train_epochs", "max_train_epochs")
    add("--network_dim", "network_dim")
    add("--output_name", "output_name")
    add("--seed", "seed")

    sched = (form.get("lr_scheduler_type") or "").strip()
    # Built-in schedulers go through --lr_scheduler; dotted-path customs through
    # --lr_scheduler_type (the resolver branch). Heuristic: a "." => custom.
    if sched:
        if "." in sched:
            extra += ["--lr_scheduler_type", sched]
        else:
            extra += ["--lr_scheduler", sched]

    for flag, key in (("--optimizer_args", "optimizer_args"),
                      ("--lr_scheduler_args", "lr_scheduler_args")):
        v = (form.get(key) or "").strip()
        if v:
            extra += [flag, *v.split()]

    if str(form.get("lr_warmup_steps", "")).strip() != "":
        extra += ["--lr_warmup_steps", str(form["lr_warmup_steps"])]

    if form.get("monitor"):
        extra.append("--monitor")
        if str(form.get("monitor_port", "")).strip() != "":
            extra += ["--monitor_port", str(form["monitor_port"])]
        if str(form.get("monitor_host", "")).strip() != "":
            extra += ["--monitor_host", str(form["monitor_host"])]

    extra_flags = (form.get("extra_flags") or "").strip()
    if extra_flags:
        extra += extra_flags.split()

    args = build_method_args(method, preset=preset, extra=extra)
    return build_launch_cmd(*args)


def launch(form: dict) -> dict:
    proc = _STATE.get("proc")
    if proc is not None and proc.poll() is None:
        return {"ok": False, "error": "A training run is already in progress."}
    cmd = build_command(form)
    if form.get("dry_run"):
        return {"ok": True, "dry_run": True, "command": " ".join(cmd)}
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"failed to spawn: {exc}"}
    _STATE.update(proc=proc, cmd=cmd, started_at=time.time())
    mon = None
    if form.get("monitor"):
        port = str(form.get("monitor_port") or "8765")
        host = str(form.get("monitor_host") or "127.0.0.1")
        shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
        mon = f"http://{shown}:{port}"
    _STATE["monitor_url"] = mon
    return {"ok": True, "command": " ".join(cmd), "pid": proc.pid, "monitor_url": mon}


def status() -> dict:
    proc = _STATE.get("proc")
    running = proc is not None and proc.poll() is None
    return {
        "running": running,
        "pid": proc.pid if proc else None,
        "returncode": (proc.poll() if proc else None) if not running else None,
        "command": " ".join(_STATE["cmd"]) if _STATE.get("cmd") else None,
        "monitor_url": _STATE.get("monitor_url"),
        "elapsed": (time.time() - _STATE["started_at"]) if _STATE.get("started_at") and running else None,
    }


def stop() -> dict:
    proc = _STATE.get("proc")
    if proc is None or proc.poll() is not None:
        return {"ok": False, "error": "no run in progress"}
    try:
        proc.terminate()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                self._send(200, HTML_FILE.read_bytes(), "text/html; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(500, str(exc).encode(), "text/plain")
        elif path == "/api/options":
            self._json(options())
        elif path == "/api/status":
            self._json(status())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        if path == "/api/launch":
            self._json(launch(body))
        elif path == "/api/stop":
            self._json(stop())
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *args):  # silence default logging
        pass


def serve(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    server = HTTPServer((host, port), Handler)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{port}"
    print(f"\n  Anima LoRA web GUI: {url}\n  (Ctrl-C to stop)\n")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  web GUI stopped.")
        server.shutdown()
