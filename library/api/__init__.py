"""Anima — programmatic front door (canonical home).

A thin façade that re-exports the handful of real entry points an embedder
needs, so driving the pipeline is "read these exports" instead of
"reverse-engineer ``inference.py`` / ``train.py`` ``main()``"::

    from library.api import generate, get_generation_settings, GenerationRequest

    settings = get_generation_settings(args)
    latent = generate(args, settings)

Each name resolves lazily (PEP 562) the first time it's accessed, so importing
this module stays cheap and avoids the circular-import chains the underlying
packages guard against.

History: this used to live at the top-level ``anima_lora`` package. It moved
here under the sd-scripts/LETS realignment (engine code lives under
``library/``); ``import anima_lora`` is kept as a back-compat lazy shim that
delegates to this module, so existing embedder scripts keep working.

The canonical homes are unchanged — this module only re-exports them:

| export | canonical home |
|--------|----------------|
| ``generate`` / ``get_generation_settings`` / ``save_output`` / ``decode_to_pil`` / ``GenerationRequest`` / ``prepare_text_inputs`` / ``ensure_text_strategies`` | ``library.inference`` |
| ``load_method_preset`` / ``read_config_from_file`` | ``library.config.io`` |
| ``load_anima_model`` | ``library.anima.weights`` |
| ``load_dit_model`` | ``library.inference.models`` |
| ``load_vae`` | ``library.models.qwen_vae`` |
| ``str_to_dtype`` | ``library.runtime.device`` |
| ``default_checkpoints`` / ``DefaultCheckpoints`` | ``library.env`` |

``ROOT`` is the repo root (the directory holding ``configs/``, ``output/`` …) as
a ``pathlib.Path`` — the single source of truth for building repo-relative paths
in tooling, instead of each script re-deriving it with its own
``Path(__file__).parents[N]`` arithmetic.

Note: repo-relative model/config paths resolve against the repo home, not the
current working directory, so importing this works from anywhere (see
``library.env.resolve_under_home`` / ``anima_home``; set ``ANIMA_HOME`` to point
at a relocated checkout).
"""

from __future__ import annotations

import importlib as _importlib
from pathlib import Path as _Path

#: Repo root — this file is ``<root>/library/api/__init__.py`` (parents[2]).
ROOT = _Path(__file__).resolve().parents[2]

# export name -> dotted module that defines it
_ATTR_TO_MODULE: dict[str, str] = {
    # generation + output (library.inference)
    "generate": "library.inference",
    "get_generation_settings": "library.inference",
    "save_output": "library.inference",
    "decode_to_pil": "library.inference",
    "GenerationRequest": "library.inference",
    "prepare_text_inputs": "library.inference",
    "ensure_text_strategies": "library.inference",
    # config merge chain (library.config.io)
    "load_method_preset": "library.config.io",
    "read_config_from_file": "library.config.io",
    # model loaders
    "load_anima_model": "library.anima.weights",
    "load_dit_model": "library.inference.models",
    "load_vae": "library.models.qwen_vae",
    # device / dtype helpers (library.runtime.device)
    "str_to_dtype": "library.runtime.device",
    # default checkpoint paths (library.env)
    "default_checkpoints": "library.env",
    "DefaultCheckpoints": "library.env",
}


def __getattr__(name: str):
    module = _ATTR_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [*_ATTR_TO_MODULE.keys(), "ROOT"]
