# -*- coding: utf-8 -*-
"""`webgui` task — start the stdlib web control panel (configure -> launch -> monitor).

Usage: ``python tasks.py webgui [--host 127.0.0.1] [--port 7860] [--no-browser]``
(env overrides: WEBGUI_HOST / WEBGUI_PORT).
"""

from __future__ import annotations

import os


def cmd_webgui(extra):
    from gui.webgui import server

    host, port, open_browser = "127.0.0.1", 7860, True
    args = list(extra or [])
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--host" and i + 1 < len(args):
            host = args[i + 1]; i += 2
        elif a == "--port" and i + 1 < len(args):
            port = int(args[i + 1]); i += 2
        elif a in ("--no-browser", "--no_browser"):
            open_browser = False; i += 1
        else:
            i += 1
    host = os.environ.get("WEBGUI_HOST", host)
    port = int(os.environ.get("WEBGUI_PORT", port))
    server.serve(host=host, port=port, open_browser=open_browser)
