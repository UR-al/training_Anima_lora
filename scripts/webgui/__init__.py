"""Web-based control panel for the merged Anima LoRA trainer.

A stdlib-only HTTP control panel (no Qt) that builds + launches training from a
browser form and links to the live web monitor. Started via
``python tasks.py webgui`` (or ``make webgui``). Distributable: depends only on
the Python stdlib + the trainer it launches.
"""
