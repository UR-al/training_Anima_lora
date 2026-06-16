# -*- coding: utf-8 -*-
"""`native-gui` task — start the PySide6 desktop control panel (→ our train.py).

Usage: ``python tasks.py native-gui``
Requires the optional ``gui`` extra — ``uv sync --extra gui`` (or
``pip install 'PySide6>=6.7'``). The PySide6 import is deferred to launch time so
the rest of tasks.py is unaffected.
"""

from __future__ import annotations


def cmd_native_gui(extra):
    try:
        from gui.native import app

        app.run()
    except ModuleNotFoundError as exc:  # PySide6 (imported in gui.native.app) missing
        if exc.name and exc.name.split(".")[0] == "PySide6":
            raise SystemExit(
                "PySide6 is not installed. Install the optional extra:\n"
                "    uv sync --extra gui\n"
                "  (or: pip install 'PySide6>=6.7')"
            ) from exc
        raise
