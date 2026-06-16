# -*- coding: utf-8 -*-
"""Native (PySide6/Qt) control panel for the Anima LoRA trainer.

A desktop GUI that drives **this repo's** ``train.py`` via the shared, torch-free
backend in :mod:`gui.backend` (``options`` / ``build_command`` / ``launch`` /
``status`` / ``stop`` / ``log_tail``) — the exact same backend the Gradio panel
uses, so both front-ends build identical commands. Native widgets give what the
web GUI can't: OS file dialogs, real tables, no localhost port / browser tab.

The web monitor (loss/LR/sample dashboard) intentionally stays web — launch it
separately (``--monitor`` / ``tools/run_monitor.py``); this panel only links to it.

Started via ``python tasks.py native-gui``. Requires the optional ``gui``
extra (``uv sync --extra gui``); the PySide6 import is deferred to launch time
so the rest of ``tasks.py`` works without it.
"""
