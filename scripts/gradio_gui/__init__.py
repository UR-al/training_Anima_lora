# -*- coding: utf-8 -*-
"""Gradio control panel for the Anima LoRA trainer.

A Gradio front-end (layout modelled on the kohya_ss GUI) that drives **this
repo's** ``train.py --method <name> --preset <name>`` rather than kohya's
``sd-scripts``. It is a thin UI shell over the proven command-builder + launch
backend in :mod:`scripts.webgui.server` (``options`` / ``build_command`` /
``launch`` / ``status`` / ``stop``) — so the dropdowns are populated from the
live registries (methods, presets, the ~89-optimizer zoo, schedulers) and the
Start button runs ``train.py`` directly as a subprocess (inline, like
``make lora``).

Started via ``python tasks.py gradio-gui`` (or ``make gradio-gui``). Requires the
optional ``gradio`` extra (``uv sync --extra gradio``); the import is lazy so the
rest of ``tasks.py`` works without it.
"""
