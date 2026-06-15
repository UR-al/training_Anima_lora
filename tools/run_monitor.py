#!/usr/bin/env python3
"""Standalone web-monitor server — the GUI's "Start monitoring" button spawns this.

The training-time monitor (``--monitor``) runs *inside* train.py for the lifetime of
the run. This entrypoint serves the SAME Chart.js dashboard at rest: it rehydrates the
last run's loss/lr/sample curves from ``library/monitoring/monitor_data/state.json`` and
serves freshly-decoded sample PNGs from ``<output_dir>/sample``. Read-only — it never
touches training — so it can run alongside a live run (give it a different port).

    python tools/run_monitor.py --host 127.0.0.1 --port 8766 --output_dir output/my_lora

The server runs on a daemon thread, so this process just rehydrates, starts it, and
sleeps to stay alive (Ctrl-C / terminate to stop).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python tools/run_monitor.py` from anywhere — import the installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.monitoring.train_monitor import (  # noqa: E402
    load_persisted_state,
    start_monitor_server,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone Anima web monitor")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument(
        "--output_dir",
        default="output",
        help="Run dir whose sample/ folder is served (e.g. output/<name>).",
    )
    ap.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the dashboard in a browser.",
    )
    args = ap.parse_args()

    # Rehydrate the last run's curves/samples/config from the on-disk state.json.
    try:
        load_persisted_state()
    except Exception as exc:  # noqa: BLE001  (best-effort; serve empty if it fails)
        print(f"[run_monitor] could not load persisted state: {exc}", file=sys.stderr)

    start_monitor_server(
        port=args.port,
        host=args.host,
        output_dir=args.output_dir,
        open_browser=not args.no_browser,
    )
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"[run_monitor] serving at http://{shown}:{args.port}  (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(3600)  # serve loop is a daemon thread — keep the process alive
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
