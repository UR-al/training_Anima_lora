"""GUI auto-preprocess → train chain runner (post-daemon, single subprocess).

The web/Gradio GUIs once chained preprocess → train through the (now-removed) job
queue. With a single blocking child process per run, this tiny runner restores the
"toggle auto-preprocess and just hit Start" flow: read a spec JSON (argv[1]), run
``tasks.py preprocess-manifest`` (with the manifest env) unless the completion
marker already matches the dataset signature, then exec the train argv. One
process, one stdout — the GUI captures it as a single run and its live-log panel
mirrors both phases.

Spec JSON::

    {"preprocess_env": {"MANIFEST_FILE": "...", "PREPROCESS_MARKER": "...",
                        "PREPROCESS_SIG": "...", ...},
     "marker": "<abs path>", "sig": "<hex>", "train_argv": ["python", "train.py", …]}
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _caches_ready(marker: str | None, sig: str | None) -> bool:
    """True iff a prior preprocess of this exact spec finished — the marker file
    exists and its stored signature matches. Mirrors server._caches_ready so the
    runner can skip a redundant preprocess on its own."""
    if not marker or not sig or not os.path.isfile(marker):
        return False
    try:
        return json.loads(Path(marker).read_text(encoding="utf-8")).get("sig") == sig
    except (OSError, ValueError):
        return False


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: gui_chain_preprocess_train.py <spec.json>", file=sys.stderr)
        return 2
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    train_argv = spec["train_argv"]

    if _caches_ready(spec.get("marker"), spec.get("sig")):
        print("=== auto-preprocess: caches present (signature match) → skipping ===",
              flush=True)
    else:
        print("=== auto-preprocess: building caches (tasks.py preprocess-manifest) ===",
              flush=True)
        env = os.environ.copy()
        env.update({k: str(v) for k, v in (spec.get("preprocess_env") or {}).items()})
        env.setdefault("PYTHONUNBUFFERED", "1")
        rc = subprocess.run(
            [sys.executable, "tasks.py", "preprocess-manifest"], env=env
        ).returncode
        if rc != 0:
            print(f"=== preprocess failed (exit {rc}) — aborting before training ===",
                  flush=True)
            return rc

    print("=== auto-preprocess done → starting training ===", flush=True)
    return subprocess.run(train_argv).returncode


if __name__ == "__main__":
    raise SystemExit(main())
