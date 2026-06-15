"""Anima programmatic front door — back-compat shim.

The façade moved to :mod:`library.api` under the sd-scripts/LETS realignment
(engine code lives under ``library/``). ``import anima_lora`` is kept working so
existing embedder scripts and ``examples/`` don't break — every attribute
resolves lazily (PEP 562) through :mod:`library.api`, so importing this stays
cheap and torch-free until a real entry point is touched.

New code should prefer ``from library.api import …`` directly.
"""

from __future__ import annotations

# Cheap re-export: library/__init__.py is empty and library/api defines only the
# lazy dispatch table + ROOT at import time (no torch), so this stays lightweight.
from library.api import ROOT  # noqa: F401


def __getattr__(name: str):
    # Delegate to library.api's own PEP-562 lazy loader so `from anima_lora import
    # generate` (etc.) resolves the canonical implementation on first access.
    import library.api as _api

    return getattr(_api, name)


def __dir__() -> list[str]:
    import library.api as _api

    return dir(_api)
