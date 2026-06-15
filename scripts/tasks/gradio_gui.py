# -*- coding: utf-8 -*-
"""`gradio-gui` task — start the Gradio control panel (kohya-style UI → our train.py).

Usage: ``python tasks.py gradio-gui [--host 127.0.0.1] [--port 7860] [--no-browser]``
(env overrides: GRADIO_GUI_HOST / GRADIO_GUI_PORT). Requires the optional
``gradio`` extra — ``uv sync --extra gradio`` (or ``pip install 'gradio>=5.34.1'``).
The gradio import is deferred to launch time so the rest of tasks.py is unaffected.
"""

from __future__ import annotations

import os


def cmd_gradio_gui(extra):
    host, port, open_browser = "127.0.0.1", 7860, True
    args = list(extra or [])
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif a == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif a in ("--no-browser", "--no_browser"):
            open_browser = False
            i += 1
        else:
            i += 1
    host = os.environ.get("GRADIO_GUI_HOST", host)
    port = int(os.environ.get("GRADIO_GUI_PORT", port))

    try:
        from gui.kohya import app

        app.serve(host=host, port=port, open_browser=open_browser)
    except ModuleNotFoundError as exc:  # gradio (imported lazily in build_app) missing
        if exc.name and exc.name.split(".")[0] == "gradio":
            raise SystemExit(
                "gradio is not installed. Install the optional extra:\n"
                "    uv sync --extra gradio\n"
                "  (or: pip install 'gradio>=5.34.1')"
            ) from exc
        raise
