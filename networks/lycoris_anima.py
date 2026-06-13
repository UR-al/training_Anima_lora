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
        "factor=4", "full_matrix=True", "enable_conv=False",
    ]

The preset file is what makes LyCORIS actually target the Anima blocks — stock
LyCORIS presets list standard diffusers class names (``Transformer2DModel`` …),
which match almost nothing in the Anima DiT. See
``configs/lycoris_presets/anima_attn_mlp.toml`` (attention+MLP, no adaln) and
``anima_full.toml`` (Block+embeds+final, includes adaln). Both pass the Anima
class names through the ``unet_target_module`` key that the kohya wrapper reads.

Everything past creation (``apply_to`` / ``prepare_optimizer_params`` /
``save_weights`` / the per-step lifecycle hooks) is the unmodified
``LycorisNetworkKohya`` instance, so it rides the generic kohya trainer path.
"""

from __future__ import annotations

import lycoris.kohya as _lyk


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
    return _lyk.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        _sanitize_te(text_encoder),
        unet,
        **kwargs,
    )


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
    """Delegate to ``lycoris.kohya.create_network_from_weights`` (TE sanitized)."""
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
