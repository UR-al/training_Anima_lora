"""Bridge: use stock ``lycoris.kohya`` (LoKr / LoHa / DyLoKr / full LyCORIS zoo)
to adapt the **Anima DiT** without forking LyCORIS.

Why this shim exists
--------------------
``train.py`` calls the network module's ``create_network`` /
``create_network_from_weights`` with the Anima calling convention, where the
"text encoder" slot is the Qwen3 stack passed as a **list that may contain
``None``** (Anima trains UNet-only and caches the TE outputs, so the live TE is
often a ``[None]`` placeholder). Our own ``networks.lora_anima`` guards against
that (``network.py``: ``if text_encoder is None: continue``), but stock
``lycoris.kohya.LycorisNetworkKohya`` does not — it does ``if text_encoder:``
(a ``[None]`` list is truthy) and then calls ``.named_modules()`` on the ``None``
element, crashing before any module is wrapped.

This module normalizes the TE argument (drop ``None`` → pass ``None`` so LyCORIS
cleanly skips the text-encoder branch) and otherwise delegates verbatim to
``lycoris.kohya``. Select it with::

    network_module = "networks.lycoris_anima"
    network_args = [
        "algo=lokr",                                       # or loha / lokr / full / ...
        "preset=configs/lycoris_presets/anima_attn_mlp.toml",
        "factor=4", "full_matrix=True",
        "train_llm_adapter=False",                         # drop the Qwen3 LLM-adapter (*_te_layers_*)
        "exclude_patterns=['.*adaln_modulation.*']",       # regex module excludes (NO spaces inside!)
    ]

The preset file is what makes LyCORIS actually target the Anima blocks — stock
LyCORIS presets list standard diffusers class names (``Transformer2DModel`` …),
which match almost nothing in the Anima DiT. See
``configs/lycoris_presets/anima_attn_mlp.toml`` (attention+MLP, no adaln) and
``anima_full.toml`` (Block+embeds+final, includes adaln). Both pass the Anima
class names through the ``unet_target_module`` key that the kohya wrapper reads.

Module exclusion (``exclude_patterns`` / ``train_llm_adapter``)
--------------------------------------------------------------
Stock ``lycoris.kohya`` targets by class/name (the preset) but has **no** regex
EXCLUDE and no ``train_llm_adapter`` knob — LoRA_Easy carried those in its own
patched ``networks/{loha,lokr}.py``, which this fork does not vendor (it uses
stock ``lycoris-lora``). We replicate them here: after creation (and before
``train.py`` calls ``apply_to``, which is what registers the modules), any wrapped
LoRA whose ``lora_name`` matches an ``exclude_patterns`` regex is dropped, plus
the Qwen3 LLM-adapter (``*_te_layers_*``) unless ``train_llm_adapter=true``
(default false, matching ``networks.lora_anima``). ``exclude_patterns`` takes a
python-literal list — **no spaces inside** (network_args are space-split):
``exclude_patterns=['.*_te_layers_.*','.*adaln_modulation.*']``.

Everything past creation (``apply_to`` / ``prepare_optimizer_params`` /
``save_weights`` / the per-step lifecycle hooks) is the unmodified
``LycorisNetworkKohya`` instance, so it rides the generic kohya trainer path.
"""

from __future__ import annotations

import logging
import re

import lycoris.kohya as _lyk

from networks.lora_anima.config import _as_bool, _as_str_list

logger = logging.getLogger(__name__)


def _apply_exclusions(network, exclude_patterns, train_llm_adapter):
    """Drop wrapped LoRA modules matching ``exclude_patterns`` (+ the LLM-adapter
    unless ``train_llm_adapter``) from the freshly-built network.

    Stock ``lycoris.kohya`` has no exclude mechanism, so we filter the module
    lists (``unet_loras`` / ``text_encoder_loras``) here — before ``apply_to``
    registers them as submodules, so excluded modules never enter the state_dict
    or optimizer. ``apply_to`` re-derives ``self.loras`` from these two lists.
    """
    patterns = list(_as_str_list(exclude_patterns) or [])
    if not _as_bool(train_llm_adapter):
        patterns.append(r".*_te_layers_.*")  # Qwen3 LLM-adapter, frozen by default
    if not patterns:
        return network
    regs = [re.compile(p) for p in patterns]

    def _excluded(m) -> bool:
        name = getattr(m, "lora_name", "") or ""
        return any(r.search(name) for r in regs)

    lists = {
        attr: getattr(network, attr, None)
        for attr in ("unet_loras", "text_encoder_loras")
    }
    total = sum(len(v) for v in lists.values() if isinstance(v, list))
    would_keep = {
        attr: [m for m in v if not _excluded(m)]
        for attr, v in lists.items()
        if isinstance(v, list)
    }
    kept_total = sum(len(v) for v in would_keep.values())

    # Safety net: a pattern that nukes EVERY module is almost always malformed
    # (classically an unquoted bracket list — exclude_patterns=[a,b] — collapsing
    # to a catch-all character-class regex). Refuse it: keep the modules and
    # surface the mistake loudly, rather than handing the optimizer an empty
    # parameter list (a confusing "optimizer got an empty parameter list" crash
    # three layers downstream).
    if total and kept_total == 0:
        logger.error(
            "lycoris_anima: exclude_patterns %s would exclude ALL %d modules — "
            "ignoring the exclusion and training the full set. Check the pattern "
            "(an unquoted bracket list like [a,b] becomes a catch-all regex).",
            patterns,
            total,
        )
        return network

    removed = total - kept_total
    for attr, kept in would_keep.items():
        setattr(network, attr, kept)
    network.loras = list(getattr(network, "text_encoder_loras", []) or []) + list(
        getattr(network, "unet_loras", []) or []
    )
    if removed:
        logger.info("lycoris_anima: excluded %d module(s) via %s", removed, patterns)
    return network


def _sanitize_te(text_encoder):
    """Drop ``None`` entries from the Anima TE slot.

    Anima hands a ``[None]`` placeholder when the TE is cached/frozen. LyCORIS
    treats a non-empty list as "wrap these" and dereferences each element, so a
    list-with-``None`` must collapse to ``None`` (skip the TE branch entirely).
    A real TE (single module or list of real modules) is passed through.
    """
    if isinstance(text_encoder, (list, tuple)):
        real = [t for t in text_encoder if t is not None]
        return real or None
    return text_encoder


def create_network(
    multiplier,
    network_dim,
    network_alpha,
    vae,
    text_encoder,
    unet,
    neuron_dropout=None,
    **kwargs,
):
    """Delegate to ``lycoris.kohya.create_network`` with the TE slot sanitized.

    ``neuron_dropout`` is accepted (train.py always passes it positionally/by
    keyword for every network) and forwarded as LyCORIS's ``dropout`` only when
    the caller did not already set one via ``network_args``.
    """
    if neuron_dropout is not None and "dropout" not in kwargs:
        kwargs["dropout"] = neuron_dropout
    # Pull the LoRA_Easy-compat exclusion knobs out before delegating — stock
    # lycoris.kohya would silently ignore them; we apply them to the built network.
    exclude_patterns = kwargs.pop("exclude_patterns", None)
    train_llm_adapter = kwargs.pop("train_llm_adapter", False)
    network = _lyk.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        _sanitize_te(text_encoder),
        unet,
        **kwargs,
    )
    return _apply_exclusions(network, exclude_patterns, train_llm_adapter)


def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    """Delegate to ``lycoris.kohya.create_network_from_weights`` (TE sanitized).

    Inference rebuilds from the saved weights, which already omit any modules
    excluded at train time — so drop the training-only exclusion knobs (no
    filtering needed here) before delegating.
    """
    kwargs.pop("exclude_patterns", None)
    kwargs.pop("train_llm_adapter", None)
    return _lyk.create_network_from_weights(
        multiplier,
        file,
        vae,
        _sanitize_te(text_encoder),
        unet,
        weights_sd=weights_sd,
        for_inference=for_inference,
        **kwargs,
    )
