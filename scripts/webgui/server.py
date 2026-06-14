# -*- coding: utf-8 -*-
"""Stdlib HTTP control panel: configure -> launch -> monitor.

Serves a single-page form whose dropdowns are populated from the live registries
(methods, presets, the ~89-optimizer zoo, schedulers), builds the exact
``train.py`` command via the shared ``scripts.tasks._common`` helpers, and spawns
training as a detached subprocess. The live loss/LR dashboard is the existing web
monitor (``--monitor``), which this panel links to.

No third-party deps — only the Python stdlib + the trainer it launches.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
HTML_FILE = Path(__file__).resolve().parent / "index.html"

# Last launch this panel issued (direct Popen and/or a daemon job id).
_STATE: dict = {
    "proc": None,
    "cmd": None,
    "monitor_url": None,
    "started_at": None,
    "daemon_job": None,
    "daemon_base": None,
}


# --------------------------------------------------------------------------- #
# Option registries (drive the form dropdowns)
# --------------------------------------------------------------------------- #
def list_methods() -> list[str]:
    d = ROOT / "configs" / "methods"
    out = sorted(p.stem for p in d.glob("*.toml")) if d.is_dir() else []
    return out or ["lora"]


def list_presets() -> list[str]:
    import tomllib

    p = ROOT / "configs" / "presets.toml"
    try:
        return list(tomllib.loads(p.read_text(encoding="utf-8")).keys()) or ["default"]
    except Exception:
        return ["default"]


def list_optimizers() -> list[str]:
    """kohya built-ins first, then the vendored zoo (class names, available only)."""
    builtins = [
        "AdamW",
        "AdamW8bit",
        "PagedAdamW8bit",
        "Lion",
        "Prodigy",
        "DAdaptAdam",
        "Adafactor",
        "RAdamScheduleFree",
        "AdamWScheduleFree",
    ]
    custom: list[str] = []
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from LoraEasyCustomOptimizer import OPTIMIZERS  # type: ignore

        custom = sorted({cls.__name__ for cls in OPTIMIZERS.values()})
    except Exception:
        pass
    seen, out = set(), []
    for name in builtins + custom:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def list_schedulers() -> list[str]:
    return [
        "cosine",
        "cosine_with_restarts",
        "constant",
        "constant_with_warmup",
        "linear",
        "polynomial",
        "warmup_stable_decay",
        "LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts",
        "LoraEasyCustomOptimizer.RexAnnealingWarmRestarts.RexAnnealingWarmRestarts",
    ]


def list_network_modules() -> list[str]:
    """Adapter backends. networks.lycoris_anima is the Anima-safe LyCORIS bridge
    (unlocks ALL LyCORIS LoRA types — LoHa/LoKr/DyLoRA/GLoRA/Full/Diag-OFT/BOFT/IA3 —
    with the torch.compile speed core intact; sanitizes Anima's cached-TE slot that
    stock lycoris.kohya crashes on). networks.lora_anima is the native adapter family.
    lycoris.kohya is the raw upstream entry (advanced / non-Anima models)."""
    mods = ["networks.lycoris_anima", "networks.lora_anima", "lycoris.kohya"]
    d = ROOT / "networks" / "methods"
    if d.is_dir():
        for p in sorted(d.glob("*.py")):
            if p.stem not in ("__init__", "base"):
                mods.append(f"networks.methods.{p.stem}")
    return mods


def list_lycoris_algos() -> list[str]:
    """LyCORIS algorithms (used with lycoris.kohya via network_args algo=...).
    The full set LoRA_Easy exposes."""
    return [
        "lora",
        "loha",
        "lokr",
        "dylora",
        "glora",
        "full",
        "diag-oft",
        "boft",
        "ia3",
    ]


# Args owned by the curated panels (don't duplicate them in the auto-generated
# "all arguments" section).
_CURATED_ARGS = {
    "method",
    "preset",
    "help",
    "optimizer_type",
    "learning_rate",
    "optimizer_args",
    "lr_scheduler",
    "lr_scheduler_type",
    "lr_scheduler_args",
    "lr_warmup_steps",
    "network_module",
    "network_args",
    "network_alpha",
    "network_dim",
    # Owned by the per-subset gradient-checkpointing toggles (the per-subset path in
    # _method_preset_extra emits --gradient_checkpointing_resolutions). Curated out so
    # it doesn't ALSO surface as a GENERAL auto-arg and emit a SECOND time — under
    # nargs="*" the later occurrence would silently clobber the per-subset union.
    "gradient_checkpointing_resolutions",
    # Preprocess-only + driven by the curated tier checkboxes (which emit --target_res);
    # inert at train time. Curate out so it doesn't also render as a BUCKET auto-arg.
    "target_res",
    "dataset_config",
    "max_train_epochs",
    "output_name",
    "output_dir",
    "logging_dir",  # derived from output_dir (<base>/<output_name>/log) in _method_preset_extra
    "resume",
    "sample_prompts",
    "seed",
    "monitor",
    "monitor_host",
    "monitor_port",
    "monitor_open_browser",
    "log_every_n_steps",  # curated in the Monitor & run panel
    # Model paths — curated in the GENERAL "Model files" controls (dit/te/vae);
    # excluded so they don't also appear as toggleable auto-args.
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
}

# Role buckets (first keyword match wins; order = display order).
# LoRA_Easy-style section taxonomy. Roles == the GUI's collapsible section keys;
# the FIRST matching keyword wins, so order + specificity matter (more-specific
# sections precede broad ones). Every introspected arg lands in one of these or
# falls through to "Other" → relabeled EXTRA (the catch-all). Curated-panel args
# (_CURATED_ARGS) are excluded entirely. Keyword choices are deliberately precise
# to avoid cross-section bleed (e.g. "lr_warmup"/"lr_decay" not bare "warmup"/
# "decay", which would grab anima's byg_/ema_ args meant for ANIMA).
_ROLE_RULES = [
    (
        "GENERAL",  # precision · seed · batch/grad · dataloader · all speed/VRAM/compile knobs
        [
            "mixed_precision", "no_half_vae", "full_bf16", "full_fp16", "fp8",
            # Global gradient_checkpointing (checkpoint EVERY block, every tier) is a
            # VRAM/speed knob, so it belongs HERE next to compile/swap/budget — NOT in
            # SAVE, whose "checkpointing" keyword used to swallow it (it landed under
            # "Save settings", which is nonsense). The subset builder's per-tier toggle
            # (--gradient_checkpointing_resolutions) is the finer-grained alternative;
            # this flag is the simple "checkpoint everything" escape hatch.
            # gradient_accumulation stays in the EXTRA catch-all (rarely needed).
            "gradient_checkpointing",
            "max_data_loader", "train_batch_size", "max_train_epochs",
            "max_train_steps", "prior_loss_weight", "lowram", "highvram",
            "compile", "dynamo", "cudagraph", "activation_memory",
            "attn_mode", "attn_softmax", "flash", "sdpa", "sageattn", "flex",
            "blocks_to_swap", "block_swap", "channel_scal",
            "persistent_data", "pin_memory", "prefetch", "dataloader",
            "split_attn", "vae_chunk", "vae_disable_cache", "vae_batch_size",
            "text_encoder_batch",  # sibling of vae_batch_size/train_batch_size (was falling to EXTRA)
            "unsloth", "cpu_offload", "fused_backward", "skip_until", "initial_",
        ],
    ),
    (
        "NETWORK",  # adapter · timestep window + flow-matching/timestep + token length · net regularization
        [
            "network",  # network_dropout / network_train_* / network_weights
            "t_min", "t_max",
            # flow-matching / timestep cluster (moved here from ANIMA per request —
            # the LoRA_Easy "Network args" tab groups these with min/max timestep).
            "timestep", "sigmoid", "weighting", "discrete_flow", "logit",
            "mode_scale", "qwen3_max_token", "t5_max_token",
            "scale_weight", "base_weights", "lora_path", "lora_multiplier",
            "dim_from_weights",
        ],
    ),
    (
        "OPTIMIZER",  # optimizer · scheduler · LR · loss
        [
            "optimizer", "unet_lr", "text_encoder_lr", "scheduler",
            "lr_warmup", "lr_decay", "loss_type", "masked_loss", "huber",
            "min_snr", "multiscale_loss", "debiased", "max_grad_norm",
            "constantcosine",  # use_constantcosine + constantcosine_tail_epochs
        ],
    ),
    # NOTE: bare "resume" was dropped — it substring-swallowed ema_resume_path (an
    # EMA knob) into SAVE, the same over-match class as the gradient_checkpointing
    # bug. --resume (the real train-state resume) is curated; resume_from_huggingface
    # is kept here via "huggingface". ema_resume_path now reaches ANIMA's "ema".
    ("SAVE", ["save", "state", "config_snapshot", "output_dir", "output_config", "metadata", "checkpointing", "huggingface"]),
    (
        "BUCKET",  # preprocessing / resolution / dataset-shape / caching toggles
        [
            "bucket", "resolution", "target_res", "resize", "min_pixels",
            "drop_lowres", "sample_ratio", "path_pattern", "dataset_repeats", "in_json",
            "use_vae_cache", "use_text_cache", "cache_info", "skip_cache",
        ],
    ),
    (
        "SUBSET",  # per-subset caption/aug knobs (the few that are also argparse)
        [
            "caption", "shuffle", "token_warmup", "reg_", "flip", "color",
            "crop", "image_dir", "keep_token", "wildcard", "alpha_mask",
            "secondary", "custom_attributes", "weighted_caption",
        ],
    ),
    ("NOISE", ["ip_noise"]),  # flow-matching input-perturbation noise (classic noise_offset N/A)
    ("SAMPLE", ["sample", "valid", "cmmd", "prompt"]),
    (
        "ANIMA",  # tokenizer path · Anima-specific experimental features (flow/timestep
        # cluster moved to NETWORK per request)
        [
            "tokenizer",  # t5_tokenizer_path / tokenizer_cache_dir
            "ema", "byg", "easycontrol", "cond_diff", "vr_", "functional",
            "llm_adapter", "self_attn_lr", "cross_attn_lr", "mlp_lr", "mod_lr",
            "artist_filter", "inversion", "use_shuffled",
        ],
    ),
]


def _arg_role(dest: str) -> str:
    for role, kws in _ROLE_RULES:
        if any(k in dest for k in kws):
            return role
    return "Other"


# Within a section the auto-args used to sort alphabetically by dest, which scatters
# related knobs (torch_compile sorts at 't', activation_memory_budget at 'a',
# dynamo_backend at 'd'...). _ARG_CLUSTERS gives each arg a SUB-GROUP label + a stable
# order, so the GUI renders related flags adjacently under a small sub-header. List
# order == display order; first keyword substring-match wins. Unmatched → no header,
# sorted last (alphabetical). Clusters are global but only surface in whatever section
# actually holds their args (the "torch.compile" cluster never shows under OPTIMIZER).
_ARG_CLUSTERS = [
    ("Precision", ["mixed_precision", "no_half_vae", "full_bf16", "full_fp16", "fp8"]),
    ("Batch & steps", ["train_batch_size", "max_train_epochs", "max_train_steps", "prior_loss_weight", "gradient_accumulation"]),
    ("torch.compile", ["torch_compile", "compile_dynamic_seq", "compile_inductor_mode", "dynamo_backend", "cudagraph", "activation_memory"]),
    ("Memory · checkpointing · offload", ["gradient_checkpointing", "blocks_to_swap", "block_swap", "cpu_offload", "unsloth", "fused_backward", "lowram", "highvram", "channel_scal"]),
    ("Attention", ["attn_mode", "attn_softmax", "flash", "sdpa", "sageattn", "flex", "split_attn"]),
    ("Dataloader", ["max_data_loader", "persistent_data", "pin_memory", "prefetch", "dataloader"]),
    ("VAE / TE encode & cache", ["vae_chunk", "vae_batch", "vae_disable", "vae_encode", "text_encoder_batch"]),
    ("Resume position", ["initial_epoch", "initial_step", "skip_until"]),
    ("Learning rate & schedule", ["unet_lr", "text_encoder_lr", "lr_scheduler", "lr_warmup", "lr_decay", "constantcosine"]),
    ("Loss", ["loss_type", "huber", "min_snr", "debiased", "masked_loss", "multiscale_loss", "max_grad_norm"]),
    ("Timestep / flow-matching", ["timestep", "sigmoid", "weighting", "discrete_flow", "logit", "mode_scale", "t_min", "t_max"]),
    ("Validation", ["validation", "validate", "cmmd", "max_validation"]),
    ("Sampling", ["sample"]),
    ("EMA", ["ema"]),
]


def _arg_cluster(dest: str) -> tuple:
    """(rank, label) sub-group for an auto-arg. Unmatched → (high rank, "") so it
    sorts last with no sub-header. See _ARG_CLUSTERS."""
    for i, (label, kws) in enumerate(_ARG_CLUSTERS):
        if any(k in dest for k in kws):
            return (i, label)
    return (len(_ARG_CLUSTERS), "")


def _arg_type(action) -> str:
    cls = action.__class__.__name__
    if cls in ("_StoreTrueAction", "_StoreFalseAction", "BooleanOptionalAction"):
        return "bool"
    t = action.type
    if t is int:
        return "int"
    if t is float:
        return "float"
    return "str"


def _jsonable(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def list_arg_groups() -> list:
    """Introspect train.py's argparse into role-grouped, toggleable arg metadata,
    so the GUI can expose EVERY --flag with its help text. Curated-panel args are
    excluded to avoid duplication."""
    try:
        import train

        parser = train.setup_parser()
    except Exception:
        return []
    buckets: dict = {}
    for a in parser._actions:
        dest = a.dest
        if dest in _CURATED_ARGS or not a.option_strings:
            continue
        # Prefer the affirmative long form. BooleanOptionalAction registers BOTH
        # --x and --no-x; --no-x is longer, so a naive max(key=len) would make the
        # toggle emit --no-x — INVERTING it (checking "use_vae_cache" would emit
        # --no-use_vae_cache, DISABLING the cache). Pick the non---no- long form.
        _affirm = [o for o in a.option_strings if not o.startswith("--no-")]
        flag = max(_affirm or a.option_strings, key=len)
        is_bool = _arg_type(a) == "bool"
        item = {
            "dest": dest,
            "flag": flag,
            "type": _arg_type(a),
            "is_bool": is_bool,
            # BooleanOptionalAction → the GUI shows a tri-state (default/on/off) so a
            # config-chain-forced-true flag can be turned OFF via --no-<flag>.
            "negatable": a.__class__.__name__ == "BooleanOptionalAction",
            "cluster": _arg_cluster(dest)[1],  # sub-group header within the section
            "default": _jsonable(a.default),
            "help": (a.help or "").strip(),
            "choices": [str(c) for c in a.choices] if a.choices else [],
            "nargs": a.nargs
            if a.nargs in ("*", "+") or isinstance(a.nargs, int)
            else None,
        }
        buckets.setdefault(_arg_role(dest), []).append(item)
    order = [r for r, _ in _ROLE_RULES] + ["Other"]
    # "Other" is the LoRA_Easy-style EXTRA catch-all (logging/metadata/console/…
    # plus anything unmatched) — the GUI's "add any --flag" bucket.
    label = {"Other": "EXTRA"}
    return [
        {
            "role": label.get(r, r),
            "args": sorted(
                buckets[r], key=lambda x: (_arg_cluster(x["dest"])[0], x["dest"])
            ),
        }
        for r in order
        if r in buckets
    ]


# Anima-specific LyCORIS presets ship as TOML files (stock built-in presets list
# standard diffusers class names that match almost nothing in the Anima DiT).
# The GUI shows the friendly name; the command builder rewrites it to the path.
_ANIMA_LYCORIS_PRESETS = {
    "anima-attn-mlp": "configs/lycoris_presets/anima_attn_mlp.toml",
    "anima-full": "configs/lycoris_presets/anima_full.toml",
}


def list_lycoris_presets() -> list[str]:
    """LyCORIS target presets (network_args preset=...) offered in the GUI.

    Anima-only: ``anima-attn-mlp`` (197 modules, attention+MLP) and ``anima-full``
    (314, +adaln/embeds) are the presets that actually wrap the Anima DiT. The stock
    LyCORIS built-ins (full / attn-only / unet-transformer-only / …) target standard
    diffusers class names absent from the Anima blocks — they wrap ~3 modules (a no-op
    run), so they're NOT offered here. _force_anima_lycoris_preset still remaps any
    stock name that arrives via import / the free network_args field, and the importer
    treats any name not in this list as "remap to anima-attn-mlp"."""
    return list(_ANIMA_LYCORIS_PRESETS)


def _force_anima_lycoris_preset(nargs: list[str]) -> list[str]:
    """Guarantee the ``networks.lycoris_anima`` bridge targets an Anima preset.

    Stock LyCORIS presets (``unet-transformer-only`` …) list standard diffusers
    class names that match **nothing** in the Anima DiT, so the network wraps 0
    modules and ``get_optimizer`` dies with "optimizer got an empty parameter
    list". A *missing* preset is just as bad (lycoris.kohya falls back to its own
    stock default). Normalize the **effective** ``preset=`` (the last one wins
    once train.py folds network_args into a kwargs dict) so the structured select,
    a stray ``preset=`` typed into the extra field, and the no-preset case are all
    covered. An explicit ``*.toml`` path (an Anima or user-supplied target file) is
    trusted and passed through.
    """
    presets = [a for a in nargs if a.startswith("preset=")]
    effective = presets[-1].split("=", 1)[1].strip() if presets else ""
    if effective.endswith(".toml"):
        return nargs
    fixed = _ANIMA_LYCORIS_PRESETS["anima-attn-mlp"]
    rest = [a for a in nargs if not a.startswith("preset=")]
    why = f"stock preset {effective!r}" if effective else "no preset"
    print(
        f"[webgui] lycoris_anima: {why} wraps no Anima modules → preset={fixed}",
        file=sys.stderr,
        flush=True,
    )
    return [f"preset={fixed}", *rest]


# kohya built-in optimizer aliases -> the real class get_optimizer constructs.
_BUILTIN_OPT_MAP = {
    "adamw8bit": "bitsandbytes.optim.AdamW8bit",
    "sgdnesterov8bit": "bitsandbytes.optim.SGD8bit",
    "lion8bit": "bitsandbytes.optim.Lion8bit",
    "pagedadamw8bit": "bitsandbytes.optim.PagedAdamW8bit",
    "pagedlion8bit": "bitsandbytes.optim.PagedLion8bit",
    "pagedadamw": "bitsandbytes.optim.PagedAdamW",
    "pagedadamw32bit": "bitsandbytes.optim.PagedAdamW32bit",
    "lion": "lion_pytorch.Lion",
    "prodigy": "prodigyopt.Prodigy",
    "dadaptadam": "dadaptation.DAdaptAdam",
    "dadaptadagrad": "dadaptation.DAdaptAdaGrad",
    "dadaptlion": "dadaptation.DAdaptLion",
    "dadaptsgd": "dadaptation.DAdaptSGD",
    "adamwschedulefree": "schedulefree.AdamWScheduleFree",
    "radamschedulefree": "schedulefree.RAdamScheduleFree",
    "sgdschedulefree": "schedulefree.SGDScheduleFree",
    "adafactor": "transformers.optimization.Adafactor",
}
# Built-in (transformers/diffusers) schedulers are functions, not classes — they
# read the trainer's --flags, not lr_scheduler_args. Curate the relevant flags.
_BUILTIN_SCHED_ARGS = {
    "cosine": ["lr_warmup_steps", "lr_scheduler_num_cycles"],
    "cosine_with_restarts": ["lr_warmup_steps", "lr_scheduler_num_cycles"],
    "cosine_with_min_lr": [
        "lr_warmup_steps",
        "lr_scheduler_num_cycles",
        "lr_scheduler_min_lr_ratio",
    ],
    "constant": [],
    "constant_with_warmup": ["lr_warmup_steps"],
    "linear": ["lr_warmup_steps"],
    "polynomial": ["lr_warmup_steps", "lr_scheduler_power"],
    "inverse_sqrt": ["lr_warmup_steps", "lr_scheduler_timescale"],
    "piecewise_constant": [],
    "warmup_stable_decay": [
        "lr_warmup_steps",
        "lr_decay_steps",
        "lr_scheduler_min_lr_ratio",
        "lr_scheduler_num_cycles",
    ],
    "adafactor": [],
}


# Short (<= 8 word) plain-language descriptions, keyed by optimizer/scheduler arg
# name. Merged into optimizer_arg_help()'s per-arg rows via _arg_desc() so the GUI
# help drawer can show "name=default — description". Optional per-optimizer
# overrides live under the "_by_opt" sub-dict (key = lowercase registry name).
# Curated (not docstring-scraped) because the vendored zoo uses inconsistent
# docstring conventions and some packages aren't even installed.
_ARG_DESCRIPTIONS = {
    # core step / learning rate
    "lr": "Base learning rate (step size).",
    "learning_rate": "Base learning rate (step size).",
    "maximize": "Maximize the objective instead of minimizing.",
    "momentum": "Fraction of previous update kept each step.",
    "nesterov": "Use Nesterov look-ahead momentum.",
    "dampening": "Damps momentum accumulation.",
    # Adam-family betas & moments
    "betas": "EMA decay rates for gradient moments.",
    "beta1": "First-moment (mean) EMA decay rate.",
    "beta2": "Second-moment (variance) EMA decay rate.",
    "beta3": "Third EMA decay (slow moment / mix).",
    "amsgrad": "Use max-of-past variance for stability.",
    "ams_bound": "AMSBound variant for tighter bounds.",
    "centered": "Normalize by centered (variance) gradient.",
    "rho": "Decay rate for squared-gradient average.",
    "alpha": "Smoothing / decay coefficient.",
    # numerical stability
    "eps": "Tiny constant preventing divide-by-zero.",
    "eps1": "Stability epsilon for the denominator.",
    "eps2": "Secondary stability epsilon term.",
    "clip_threshold": "Clamp on RMS of the update.",
    # weight decay
    "weight_decay": "L2 / weight-decay regularization strength.",
    "weight_decouple": "Decouple weight decay like AdamW.",
    "decouple": "Decouple weight decay from gradient.",
    "fixed_decay": "Use a fixed (un-scaled) weight decay.",
    "cautious_weight_decay": "Decay only sign-aligned coordinates.",
    # Prodigy / D-Adaptation auto-LR
    "d_coef": "Multiplier on the estimated learning rate.",
    "d0": "Initial learning-rate estimate.",
    "use_bias_correction": "Apply Adam-style bias correction.",
    "bias_correction": "Apply Adam-style bias correction.",
    "safeguard_warmup": "Stabilize LR estimate during warmup.",
    # update strategy / cautious / schedule-free
    "cautious": "Mask updates conflicting with the gradient.",
    "update_strategy": "Update-masking mode (cautious / grams / etc.).",
    "r": "Schedule-free polynomial weighting power.",
    "weight_lr_power": "LR weighting exponent for averaging.",
    # low-precision / performance
    "foreach": "Batch ops across params for speed.",
    "fused": "Use a fused CUDA kernel.",
    "kahan_sum": "Kahan compensation for low-precision updates.",
    "stochastic_rounding": "Stochastic rounding for bf16/fp16 updates.",
    "compile_step": "torch.compile the per-parameter step.",
    # factored / memory-efficient state
    "factored": "Factorize second moment to save memory.",
    "non_factored_confidence": "Apply confidence term to 1D params.",
    "slice_p": "Subsample params for LR estimation.",
    "sync_chunk_size": "Chunk size for cross-device state sync.",
    "state_storage_dtype": "Dtype for stored optimizer state.",
    "state_storage_device": "Device holding optimizer state (e.g. cpu).",
    # misc shared
    "gamma": "Per-step decay / shrink factor.",
    "growth_rate": "Cap on per-step learning-rate growth.",
    "warmup_steps": "Steps to linearly ramp up.",
    "warmup_init": "Start warmup from a tiny LR.",
    # scheduler args
    "num_cycles": "Number of cosine restart cycles.",
    "power": "Polynomial decay exponent.",
    "min_lr": "Lower bound on the learning rate.",
    "min_lr_ratio": "Floor LR as a fraction of peak.",
    "restart_decay": "LR scale applied after each restart.",
    "warmup_ratio": "Fraction of training spent warming up.",
    "first_cycle_steps": "Length of the first restart cycle.",
    "first_cycle_max_steps": "Length of the first restart cycle.",
    "cycle_mult": "Per-cycle length multiplier.",
    "cycle_multiplier": "Per-cycle length multiplier (CAWR/RAWR restart growth).",
    "max_lr": "Peak learning rate at cycle start.",
    "d": "Rex schedule shape parameter.",
    # per-optimizer overrides (key = lowercase friendly name OR a dotted path's
    # class-name tail — see _arg_desc; e.g. "came" matches both `CAME` and
    # `LoraEasyCustomOptimizer.came.CAME` / `pytorch_optimizer.…came.CAME`).
    "_by_opt": {
        "adopt": {"clip": "Gradient clip; ADOPT default 0.25."},
        "scion": {"gamma": "Norm/constraint scaling for Scion LMO."},
        "came": {
            # CAME takes THREE betas (Adam takes two): grad EMA, grad² EMA, and
            # the instability/confidence EMA — the term that down-weights noisy
            # updates. Vendored LoraEasyCustomOptimizer.came (friendly `CAME`,
            # lr≈5e-5) adds cautious/grams update_strategy + CPU state offload
            # (state_storage_*); pytorch_optimizer's CAME (lr≈2e-4) is leaner +
            # has `maximize`; `customized_optimizers.came` is legacy/not installed.
            "betas": "3 EMA rates: grad · grad² · instability (CAME uses 3, not 2).",
            "clip_threshold": "RMS clip on the update vector (CAME; default 1.0).",
            "eps1": "Denominator stability ε for the 2nd moment (CAME).",
            "eps2": "Stability ε for the instability/confidence term (CAME).",
        },
    },
}


def _arg_desc(arg_name: str, opt_key: str | None = None) -> str | None:
    """Short description for an optimizer/scheduler arg (None if unknown).
    Per-optimizer override (``_by_opt[opt_key][arg]``) wins over the flat map."""
    a = (arg_name or "").lstrip("-").lower()
    if opt_key:
        byo = _ARG_DESCRIPTIONS.get("_by_opt", {})
        k = opt_key.lower()
        # Match both the friendly name ("came") AND a dotted path's class-name
        # tail ("loraeasycustomoptimizer.came.came" / "pytorch_optimizer.…came.came"
        # → "came"), so per-optimizer overrides fire however the optimizer_type
        # was spelled — the same CAME exists in three packages.
        for cand in (k, k.rsplit(".", 1)[-1]):
            ov = byo.get(cand)
            if ov and a in ov:
                return ov[a]
    return _ARG_DESCRIPTIONS.get(a)


def optimizer_arg_help(name: str) -> dict:
    """What an optimizer/scheduler accepts, for the GUI help drawer. Introspects the
    class __init__ for custom/3rd-party optimizers; resolves kohya built-in aliases
    to their real class; and for built-in schedulers (functions, not classes) lists
    the trainer --flags they read. Each arg row carries a short ``desc``."""
    import importlib
    import inspect

    key = name.lower()
    if key in _BUILTIN_SCHED_ARGS:  # built-in scheduler -> trainer flags
        flags = _BUILTIN_SCHED_ARGS[key]
        note = "configured via the main flags (see All arguments): " + (
            ", ".join("--" + f for f in flags) if flags else "no extra args needed"
        )
        return {
            "ok": True,
            "builtin_scheduler": True,
            "note": note,
            "args": [
                {
                    "name": "--" + f,
                    "default": None,
                    "required": False,
                    "desc": _arg_desc(
                        f.replace("lr_scheduler_", "").replace("lr_", "")
                    ),
                }
                for f in flags
            ],
        }

    cls = None
    try:
        if "." in name:
            vals = name.split(".")
            cls = getattr(importlib.import_module(".".join(vals[:-1])), vals[-1])
        else:
            from LoraEasyCustomOptimizer import OPTIMIZERS

            cls = OPTIMIZERS.get(key)
            if cls is None and key in _BUILTIN_OPT_MAP:
                vals = _BUILTIN_OPT_MAP[key].split(".")
                cls = getattr(importlib.import_module(".".join(vals[:-1])), vals[-1])
            if cls is None:
                import torch

                cls = getattr(torch.optim, name, None)
    except Exception:
        cls = None
    if cls is None:
        return {
            "ok": False,
            "args": [],
            "note": "no introspectable args (its package may not be installed)",
        }
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return {"ok": False, "args": []}
    args = []
    for pn, p in sig.parameters.items():
        if pn in (
            "self",
            "params",
            "model",
            "optimizer",
            "base_optimizer",
        ) or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        empty = p.default is inspect.Parameter.empty
        args.append(
            {
                "name": pn,
                "default": None if empty else _jsonable(p.default),
                "required": empty,
                "desc": _arg_desc(pn, key),
            }
        )
    return {"ok": True, "cls": f"{cls.__module__}.{cls.__qualname__}", "args": args}


# Constant-token resolution tiers (the honest anima equivalent of kohya's
# min/max bucket reso + steps — anima has no per-image bucket knobs; each tier
# edge maps to a curated token-count bucket family). Source of truth:
# library.datasets.buckets.ALLOWED_TARGET_RES.
def list_target_res_tiers() -> list[int]:
    try:
        from library.datasets.buckets import ALLOWED_TARGET_RES

        return sorted(int(e) for e in ALLOWED_TARGET_RES)
    except Exception:
        return [512, 768, 896, 1024, 1280, 1536]


def _sam3_available() -> bool:
    """Whether SAM3 masking can run (its gated weights are present)."""
    d = ROOT / "models" / "sam3"
    return (d / "sam3.pt").exists() or (d / "config.json").exists()


def _mit_available() -> bool:
    return (ROOT / "models" / "mit" / "model.pth").exists()


_OPTIONS_CACHE = None


def options() -> dict:
    """Cached option registries. The first call imports the ~89-optimizer zoo
    (slow); cache it so page reloads are instant. Pre-warmed in serve()."""
    global _OPTIONS_CACHE
    if _OPTIONS_CACHE is None:
        _OPTIONS_CACHE = {
            "methods": list_methods(),
            "presets": list_presets(),
            "optimizers": list_optimizers(),
            "schedulers": list_schedulers(),
            "network_modules": list_network_modules(),
            "lycoris_algos": list_lycoris_algos(),
            "lycoris_presets": list_lycoris_presets(),
            "arg_groups": list_arg_groups(),
            "target_res_tiers": list_target_res_tiers(),
            "sam3_available": _sam3_available(),
            "mit_available": _mit_available(),
        }
    return _OPTIONS_CACHE


# --------------------------------------------------------------------------- #
# Command building + launch
# --------------------------------------------------------------------------- #
def _method_preset_extra(form: dict):
    """(method, preset, extra) from the form — shared by the preview, the direct
    Popen path, and the daemon-submit path."""
    method = (form.get("method") or "lora").strip()
    preset = (form.get("preset") or "default").strip()
    # A LyCORIS network can't ride the `lora` method (it carries native-adapter
    # flags — ortho/timestep_mask/llm-adapter caching — meant for networks.lora_anima).
    # When the user picks a lycoris backend but left the method at the `lora`
    # default, route to the clean `lycoris` method so the run "just works".
    if "lycoris" in (form.get("network_module") or "") and method == "lora":
        if (ROOT / "configs" / "methods" / "lycoris.toml").is_file():
            method = "lycoris"
    extra: list[str] = []

    def add(flag: str, key: str) -> None:
        v = form.get(key)
        if v not in (None, "", []):
            extra.extend([flag, str(v)])

    add("--optimizer_type", "optimizer_type")
    add("--learning_rate", "learning_rate")
    add("--dataset_config", "dataset_config")
    add("--max_train_epochs", "max_train_epochs")
    add("--network_dim", "network_dim")
    # Output layout: everything for a run lives under <base>/<output_name>/ —
    # the <output_name>.safetensors checkpoint at the top, sample/ (auto-made by
    # the trainer at <output_dir>/sample) and log/ (TensorBoard) one level inside.
    # <base> is the form's output_dir field (default "output", repo-relative);
    # <output_name> defaults to the method name when the field is blank. We always
    # emit --output_name so the checkpoint filename matches its folder.
    eff_name = (form.get("output_name") or "").strip() or method or "anima_lora"
    out_base = (form.get("output_dir") or "output").strip().rstrip("/\\ ") or "output"
    out_dir = f"{out_base}/{eff_name}"
    extra += [
        "--output_name",
        eff_name,
        "--output_dir",
        out_dir,
        "--logging_dir",
        f"{out_dir}/log",
    ]
    add("--resume", "resume")
    add("--sample_prompts", "sample_prompts")
    add("--seed", "seed")
    # Metric/monitor/cmd-progress cadence: log every N optimizer steps.
    add("--log_every_n_steps", "log_every_n_steps")

    # Model paths (DiT / text-encoder / VAE). Blank = config-chain default
    # (models/…). Set them to point at forge-neo / ComfyUI model files so the
    # repo needs no local weights — the same paths ride the auto-preprocess
    # CONFIG_FILE snapshot too (see _prepare_auto_preprocess).
    add("--pretrained_model_name_or_path", "dit_path")
    add("--qwen3", "te_path")
    add("--vae", "vae_path")

    sched = (form.get("lr_scheduler_type") or "").strip()
    # Built-in schedulers go through --lr_scheduler; dotted-path customs through
    # --lr_scheduler_type (the resolver branch). Heuristic: a "." => custom.
    if sched:
        if "." in sched:
            extra += ["--lr_scheduler_type", sched]
        else:
            extra += ["--lr_scheduler", sched]

    for flag, key in (
        ("--optimizer_args", "optimizer_args"),
        ("--lr_scheduler_args", "lr_scheduler_args"),
    ):
        v = (form.get(key) or "").strip()
        if v:
            extra += [flag, *v.split()]

    if str(form.get("lr_warmup_steps", "")).strip() != "":
        extra += ["--lr_warmup_steps", str(form["lr_warmup_steps"])]

    if form.get("monitor"):
        extra.append("--monitor")
        if str(form.get("monitor_port", "")).strip() != "":
            extra += ["--monitor_port", str(form["monitor_port"])]
        if str(form.get("monitor_host", "")).strip() != "":
            extra += ["--monitor_host", str(form["monitor_host"])]

    # Adapter / LoRA type: network_module (e.g. lycoris.kohya for the LyCORIS
    # algos — LoRA/LoHa/LoKr/DyLoRA/GLoRA/Full/Diag-OFT/BOFT — or networks.lora_anima
    # for the native adapters) + algo (folded into network_args) + alpha + free args.
    nm = (form.get("network_module") or "").strip()
    # Stock lycoris.kohya wraps almost nothing on the Anima DiT — its presets list
    # diffusers class names, so the Anima blocks miss and only ~3 FinalLayer modules
    # get wrapped (a no-op run) — and it crashes on Anima's [None] TE slot. Any
    # LyCORIS module must ride the Anima bridge, which sanitizes the TE and pairs
    # with the anima_* presets that actually wrap the blocks. Mirror the config
    # importer's routing so a stale form / non-anima default can't ship a 3-module
    # run. (Also makes the preset guard below fire — it keys on "lycoris_anima".)
    if nm and "lycoris" in nm and nm != "networks.lycoris_anima":
        print(
            f"[webgui] network_module {nm!r} → networks.lycoris_anima "
            "(stock LyCORIS wraps ~3 Anima modules; the bridge wraps the blocks).",
            file=sys.stderr,
            flush=True,
        )
        nm = "networks.lycoris_anima"
    if nm:
        extra += ["--network_module", nm]
    na = str(form.get("network_alpha", "")).strip()
    if na:
        extra += ["--network_alpha", na]
    _na = form.get("network_args") or ""
    # Honor shell-style quotes so a value with spaces survives as one token
    # (e.g. caption="a b"), but only when quotes are actually present — the plain
    # `key=val key=val` case stays byte-identical to .split(), and Windows
    # backslash paths (never quoted) are left untouched.
    nargs = shlex.split(_na, posix=True) if ('"' in _na or "'" in _na) else _na.split()
    if "lycoris" in nm:  # lycoris.kohya OR networks.lycoris_anima (the Anima bridge)
        lp = (form.get("lycoris_preset") or "").strip()
        if lp:
            # Friendly Anima preset names → shipped TOML paths; pass others verbatim.
            lp = _ANIMA_LYCORIS_PRESETS.get(lp, lp)
            nargs = [f"preset={lp}"] + nargs
        algo = (form.get("algo") or "").strip()
        if algo:
            nargs = [f"algo={algo}"] + nargs
        # The Anima bridge MUST target an Anima preset (a stock built-in name — or
        # no preset — wraps 0 DiT modules → "optimizer got an empty parameter
        # list"). Normalize whatever rode in from the select / extra field.
        if "lycoris_anima" in nm:
            nargs = _force_anima_lycoris_preset(nargs)
    if nargs:
        extra += ["--network_args", *nargs]

    # Per-subset gradient checkpointing → --gradient_checkpointing_resolutions: the
    # union of tier edges of subsets whose checkpointing toggle is on (a tier
    # checkpoints if ANY subset using it has it checked). Blank tiers on a checked
    # subset = "all", so fall back to the globally-enabled target_res tiers. Lets a
    # big tier (e.g. 1536) fit via checkpointing while the smaller tiers stay
    # full-speed — no global gradient_checkpointing / block-swap / budget needed.
    _global_tiers = [int(t) for t in (form.get("target_res") or []) if str(t).strip()]
    _gc_edges: set = set()
    for s in form.get("subsets") or []:
        if not s.get("gradient_checkpointing"):
            continue
        ts = [int(x) for x in re.findall(r"\d+", str(s.get("tiers") or ""))]
        _gc_edges.update(ts or _global_tiers)
    if _gc_edges:
        extra += [
            "--gradient_checkpointing_resolutions",
            *[str(e) for e in sorted(_gc_edges)],
        ]

    # Auto-generated "all arguments" section: each enabled item is
    # {flag, value, is_bool, nargs}. Only enabled args are emitted (the rest fall
    # back to the config-chain defaults).
    for item in form.get("adv") or []:
        flag = item.get("flag")
        if not flag:
            continue
        # Negatable bool (BooleanOptionalAction) → tri-state. "default" defers to the
        # config chain (emit nothing); "on" emits the affirmative; "off" emits
        # --no-<flag> to force it false even when base.toml/preset set it true (the
        # stuck-on-bool fix — e.g. --no-compile_dynamic_seq for static compile).
        if item.get("negatable"):
            tri = item.get("tri") or "default"
            if tri == "on":
                extra.append(flag)
            elif tri == "off":
                extra.append("--no-" + flag[2:])  # --torch_compile → --no-torch_compile
            continue
        # `on` is the toggle: the GUI now saves typed-but-unchecked auto-args too
        # (for round-trip), so skip any explicitly toggled OFF. Missing `on`
        # (imported configs / older saves) defaults to enabled.
        if item.get("on", True) is False:
            continue
        if item.get("is_bool"):
            if item.get("value"):
                # Self-heal stale negated flags from configs saved/imported before
                # the --no- inversion fix: a BooleanOptionalAction's affirmative is
                # the flag minus its auto-generated "--no-" prefix (hyphen, so a
                # real "--no_half_vae" with an underscore is left untouched). An
                # enabled toggle must emit --use_text_cache, never --no-use_text_cache.
                if flag.startswith("--no-"):
                    flag = "--" + flag[len("--no-") :]
                extra.append(flag)
        else:
            val = item.get("value")
            if val not in (None, "", []):
                if item.get("nargs") in ("*", "+"):
                    extra += [flag, *str(val).split()]
                else:
                    extra += [flag, str(val)]

    extra_flags = (form.get("extra_flags") or "").strip()
    if extra_flags:
        extra += extra_flags.split()

    return method, preset, extra


def build_command(form: dict) -> list[str]:
    """The exact train.py launch command (preview / direct-Popen path)."""
    from scripts.tasks._common import build_launch_cmd, build_method_args

    method, preset, extra = _method_preset_extra(form)
    return build_launch_cmd(*build_method_args(method, preset=preset, extra=extra))


def _monitor_url(form: dict):
    if not form.get("monitor"):
        return None
    port = str(form.get("monitor_port") or "8766")
    host = str(form.get("monitor_host") or "127.0.0.1")
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    return f"http://{shown}:{port}"


def _dataset_subsets(form: dict) -> list[dict]:
    """Normalize the GUI subset list (or the legacy single raw_image_dir + ds_*
    fields) into clean subset dicts: image_dir [+ cache_dir] + per-subset settings.
    This is the single dataset definition both the auto-preprocess and pre-cached
    paths consume."""
    import re

    out: list[dict] = []
    for s in form.get("subsets") or []:
        if not isinstance(s, dict):
            continue
        img = str(s.get("image_dir") or "").strip()
        if not img:
            continue
        d: dict = {"image_dir": img}
        if str(s.get("cache_dir") or "").strip():
            d["cache_dir"] = str(s["cache_dir"]).strip()
        for k, cast in (("num_repeats", int), ("keep_tokens", int),
                        ("caption_extension", str), ("caption_dropout_rate", float),
                        ("batch_size", int), ("random_crop_padding_percent", float)):
            v = s.get(k)
            if v in (None, ""):
                continue
            try:
                d[k] = cast(v)
            except (TypeError, ValueError):
                pass
        for fk in ("flip_aug", "random_crop"):
            if s.get(fk):
                d[fk] = True
        # per-subset tiers (multi-scale): "512,1024" → [512,1024]; blank = all tiers
        tlist = [int(x) for x in re.findall(r"\d+", str(s.get("tiers") or ""))]
        if tlist:
            d["tiers"] = tlist
        out.append(d)
    if not out:  # back-compat: a single raw folder + the panel-level ds_* fields
        raw = str(form.get("raw_image_dir") or "").strip()
        if raw:
            d = {"image_dir": raw}
            if str(form.get("ds_num_repeats") or "").strip():
                d["num_repeats"] = int(form["ds_num_repeats"])
            if str(form.get("ds_keep_tokens") or "").strip():
                d["keep_tokens"] = int(form["ds_keep_tokens"])
            if str(form.get("ds_caption_extension") or "").strip():
                d["caption_extension"] = form["ds_caption_extension"]
            cdr = str(form.get("ds_caption_dropout_rate") or "").strip()
            if cdr:
                try:
                    d["caption_dropout_rate"] = float(cdr)
                except ValueError:
                    pass
            if form.get("ds_flip_aug"):
                d["flip_aug"] = True
            if form.get("ds_random_crop"):
                d["random_crop"] = True
            out.append(d)
    return out


def _build_precached_config(form: dict) -> str | None:
    """Auto-preprocess OFF → build a dataset config from the subsets treated as
    already resized + cached (image_dir + cache_dir as given). Returns its path."""
    import toml as _toml

    subs = _dataset_subsets(form)
    if not subs:
        return None
    bs = int(form.get("ds_batch") or 1)
    keep = ("image_dir", "cache_dir", "num_repeats", "keep_tokens",
            "caption_extension", "caption_dropout_rate", "flip_aug", "random_crop")
    blocks = []
    for s in subs:
        blk = {k: s[k] for k in keep if k in s}
        blk.setdefault("recursive", True)
        blocks.append({"batch_size": int(s.get("batch_size") or bs), "subsets": [blk]})
    name = _safe_name(form.get("ds_name") or "dataset")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASET_DIR / f"{name}_precached.toml"
    path.write_text(_toml.dumps({"datasets": blocks}), encoding="utf-8")
    return str(path)


def _dataset_fingerprint(entries) -> str:
    """A fast signature of an auto-preprocess run: the manifest entries (all
    settings — tiers, min_pixels, random_crop, cache/resized dirs) PLUS a
    stat-only fingerprint of every source image (relpath, size, mtime). No image
    decode — just os.walk + os.stat — so it's cheap to recompute each launch.
    Changing a setting OR adding/removing/editing an image flips the hash.
    """
    import hashlib
    import json as _json

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif")
    parts = [_json.dumps(entries, sort_keys=True, default=str)]
    for src in sorted({e.get("src") for e in entries if e.get("src")}):
        files = []
        try:
            for root, _dirs, names in os.walk(src):
                for n in names:
                    if n.lower().endswith(exts):
                        fp = os.path.join(root, n)
                        try:
                            stt = os.stat(fp)
                            rel = os.path.relpath(fp, src).replace("\\", "/")
                            files.append((rel, stt.st_size, int(stt.st_mtime)))
                        except OSError:
                            pass
        except OSError:
            pass
        files.sort()
        parts.append(src + "::" + repr(files))
    return hashlib.sha1("\n".join(parts).encode("utf-8", "replace")).hexdigest()


def _caches_ready(prep: dict) -> bool:
    """True iff a prior preprocess of this EXACT spec finished — i.e. the marker
    file exists, its stored signature matches the current one, and the cache root
    is still present. Lets launch() skip the preprocess job and train directly.
    """
    marker, sig = prep.get("_marker"), prep.get("_sig")
    if not marker or not sig:
        return False
    path = marker if os.path.isabs(marker) else str(ROOT / marker)
    if not os.path.isfile(path) or not os.path.isdir(os.path.dirname(path)):
        return False
    try:
        import json as _json

        return _json.loads(open(path, encoding="utf-8").read()).get("sig") == sig
    except (OSError, ValueError):
        return False


def _prepare_auto_preprocess(form: dict) -> dict:
    """Set up the auto-preprocess→train daemon chain from the form.

    Resizes/caches (and optionally masks) a RAW image folder at training start —
    like latent caching, but kicked off automatically. Returns the command-job
    spec (``argv`` + ``extra_env``) for ``submit_command``; the caller chains the
    train job after it via ``chain_train``. Also REWRITES ``form['dataset_config']``
    to the auto-generated config pointing at the to-be-created cache dirs, so the
    chained train job reads the fresh caches. Returns ``{"error": …}`` on a bad
    folder.

    Mechanism: a CONFIG_FILE snapshot redirects source/resized/cache/mask dirs
    (+ target_res tiers) so the standard ``preprocess`` / ``mask`` pipeline runs
    against the user's folder without editing configs/. Both preprocess and mask
    read these path overrides; masking self-skips via RUN_SAM_MASK/RUN_MIT_MASK.
    """
    import json as _json
    import toml as _toml

    subs = _dataset_subsets(form)
    if not subs:
        return {"error": "Auto-preprocess is on but no dataset subset / image folder is set."}
    for s in subs:
        if not Path(s["image_dir"]).is_dir():
            return {"error": f"Image folder not found: {s['image_dir']}"}

    # Everything generated for this run lives under cache/<output_name>/: the
    # VAE/TE/PE caches (split into vae/te/pe subfolders automatically by
    # resolve_cache_path under each subset's cache_dir), plus resized/ and
    # masks/. Keyed on output_name so a run's data is self-contained per the
    # requested layout.
    name = _safe_name(form.get("output_name") or form.get("ds_name") or "gui")
    base_resized = f"cache/{name}/resized"
    base_cache = f"cache/{name}"
    base_mask = f"cache/{name}/masks"
    masking = bool(form.get("mask_enable"))
    tiers = sorted(int(t) for t in (form.get("target_res") or []) if str(t).strip()) or [1024]
    multiscale = bool(form.get("multiscale")) and len(tiers) >= 2
    bs = int(form.get("ds_batch") or 1)

    # per-tier skip edges (multi-scale): explicit ms_skip "tier:edge,…", else auto
    # (next-lower tier), unless "skip upscaling" is off (force every image in).
    skip_map: dict[int, int] = {}
    for part in str(form.get("ms_skip") or "").split(","):
        if ":" in part:
            tk, sk = part.split(":", 1)
            try:
                skip_map[int(tk)] = int(sk)
            except ValueError:
                pass
    no_skip = form.get("ms_skip_upscale") is False

    def _skip_minpx(tier: int) -> int:
        # skip edge keyed by the tier's place in the GLOBAL tier list (so per-subset
        # tier choices still get the right next-lower-tier auto threshold).
        if no_skip:
            return 0
        if tier in skip_map:
            edge = skip_map[tier]
        else:
            gi = tiers.index(tier) if tier in tiers else 0
            edge = 0 if gi == 0 else tiers[gi - 1]
        return edge * edge

    entries: list[dict] = []
    datasets: list[dict] = []
    for i, s in enumerate(subs):
        block_common = {k: s[k] for k in ("num_repeats", "keep_tokens", "caption_extension", "caption_dropout_rate") if s.get(k) not in (None, "")}
        if s.get("flip_aug"):
            block_common["flip_aug"] = True
        # random_crop is BAKED into the resized PNG at preprocess time (it can't act
        # on the fixed cached latents training reads), so it rides the resize ENTRY,
        # not the training subset block.
        rc_entry: dict = {}
        if s.get("random_crop"):
            rc_entry["random_crop"] = True
            if s.get("random_crop_padding_percent") not in (None, ""):
                rc_entry["random_crop_padding_percent"] = s["random_crop_padding_percent"]
        sub_bs = int(s.get("batch_size") or bs)  # per-subset batch, else dataset default
        # per-subset tiers (multi-scale) — intersect with the globally-enabled tiers;
        # blank falls back to all of them. Lets a subset target one resolution with
        # its own batch/repeat (kohya per-block parity).
        sub_tiers = [t for t in (s.get("tiers") or tiers) if t in tiers] or tiers
        if multiscale:
            for t in sub_tiers:
                rdir, cdir = f"{base_resized}/{i}/{t}", f"{base_cache}/{i}/{t}"
                mdir = f"{base_mask}/{i}/{t}" if masking else None
                e = {"src": s["image_dir"], "resized": rdir, "cache": cdir,
                     "target_res": str(t), "min_pixels": _skip_minpx(t), **rc_entry}
                if mdir:
                    e["mask"] = mdir
                entries.append(e)
                blk = {"image_dir": rdir, "cache_dir": cdir, "recursive": True, **block_common}
                if mdir:
                    blk["mask_dir"] = mdir
                datasets.append({"batch_size": sub_bs, "subsets": [blk]})
        else:
            rdir, cdir = f"{base_resized}/{i}", f"{base_cache}/{i}"
            mdir = f"{base_mask}/{i}" if masking else None
            e = {"src": s["image_dir"], "resized": rdir, "cache": cdir,
                 "target_res": " ".join(str(t) for t in sub_tiers),
                 "min_pixels": 0 if form.get("drop_lowres") is False else 500000,
                 **rc_entry}
            if mdir:
                e["mask"] = mdir
            entries.append(e)
            blk = {"image_dir": rdir, "cache_dir": cdir, "recursive": True, **block_common}
            if mdir:
                blk["mask_dir"] = mdir
            datasets.append({"batch_size": sub_bs, "subsets": [blk]})

    # training dataset config → the (to-be-created) cache dirs
    ds_toml = _toml.dumps({"datasets": datasets})
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ds_path = DATASET_DIR / f"{name}_auto.toml"
    ds_path.write_text(ds_toml, encoding="utf-8")
    form["dataset_config"] = str(ds_path)

    # preprocess manifest (one entry per subset × tier)
    manifest: dict = {
        "caption_shuffle_variants": str(form.get("caption_shuffle_variants") or "4"),
        "caption_tag_dropout_rate": str(form.get("caption_tag_dropout_rate") or "0.1"),
        "entries": entries,
    }
    for mk, fk in (("vae", "vae_path"), ("qwen3", "te_path"), ("dit", "dit_path")):
        v = (form.get(fk) or "").strip()
        if v:
            manifest[mk] = v
    # REPA v2: if the form's network_args carry use_repa, tell the manifest loop
    # to also cache PE-Spatial features (so the chained train job finds them).
    na_kv = dict(
        tok.split("=", 1)
        for tok in str(form.get("network_args") or "").split()
        if "=" in tok
    )
    if na_kv.get("use_repa", "").strip().lower() in ("1", "true", "yes"):
        manifest["use_repa"] = True
        manifest["repa_encoder"] = na_kv.get("repa_encoder") or "pe_spatial"
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    mf_path = STORE_DIR / f"manifest_{name}.json"
    mf_path.write_text(_json.dumps(manifest, indent=2), encoding="utf-8")

    # Completion marker + signature: the preprocess task writes `marker` (under the
    # cache root) containing `sig` on full success; a later launch with the SAME
    # sig (settings + source files unchanged) skips preprocess and trains directly.
    sig = _dataset_fingerprint(entries)
    # Absolute so the daemon (which writes it from its OWN cwd) and the GUI (which
    # reads it via _caches_ready) resolve to the SAME file even across checkouts —
    # a relative path silently diverges and forces a needless re-preprocess.
    marker = str(ROOT / base_cache / ".anima_preprocess.json")
    env = {
        "MANIFEST_FILE": str(mf_path),
        "PREPROCESS_MARKER": marker,
        "PREPROCESS_SIG": sig,
    }
    if masking:
        env["RUN_SAM_MASK"] = "1" if form.get("mask_sam") else "0"
        env["RUN_MIT_MASK"] = "1" if form.get("mask_mit") else "0"
        if str(form.get("mit_text_threshold") or "").strip():
            env["MIT_TEXT_THRESHOLD"] = str(form["mit_text_threshold"])
        if str(form.get("mit_dilate") or "").strip():
            env["MIT_DILATE"] = str(form["mit_dilate"])
        if form.get("mask_sam") and str(form.get("sam3_path") or "").strip():
            env["SAM3_CHECKPOINT"] = str(form["sam3_path"]).strip()
    return {
        "argv": ["tasks.py", "preprocess-manifest"],
        "extra_env": env,
        "dataset_config": str(ds_path),
        "training_toml": ds_toml,
        "masking": masking,
        "multiscale": multiscale,
        "target": "preprocess-manifest",
        "_marker": marker,
        "_sig": sig,
    }


def _autobatch_argv(form: dict):
    """Build the ``tasks.py bench-autobatch`` argv from the self-contained ab_* GUI
    fields. Returns (argv, error): error is non-None when nothing to search."""
    res = [str(int(r)) for r in (form.get("ab_res") or []) if str(r).strip()]
    if not res:
        return None, "체크된 해상도가 없습니다 — search할 해상도를 고르세요."
    gc_res = [str(int(r)) for r in (form.get("ab_gradckpt_res") or []) if str(r).strip()]
    argv = ["tasks.py", "bench-autobatch", "--res", *res,
            "--max-batch", str(int(form.get("ab_max_batch") or 8))]
    if gc_res:
        argv += ["--gradient_checkpointing_resolutions", *gc_res]
    nm = (form.get("ab_network_module") or "networks.lora_anima").strip()
    # Same routing as the training builder: stock lycoris.kohya wraps ~3 Anima
    # modules, so the bench would measure a no-op. Force the Anima bridge.
    if nm and "lycoris" in nm and nm != "networks.lycoris_anima":
        nm = "networks.lycoris_anima"
    argv += ["--network_module", nm,
             "--network_dim", str(int(form.get("ab_network_dim") or 16)),
             "--network_alpha", str(form.get("ab_network_alpha") or 8.0)]
    na = str(form.get("ab_network_args") or "").strip()
    # quote-aware split (same as the training network_args path)
    nargs = (shlex.split(na, posix=True) if ('"' in na or "'" in na) else na.split()) if na else []
    if "lycoris_anima" in nm:
        # ensure an Anima preset even if ab_network_args is blank / has a stock preset
        nargs = _force_anima_lycoris_preset(nargs)
    if nargs:
        argv += ["--network_args", *nargs]
    argv += ["--optimizer_type", (form.get("ab_optimizer_type") or "AdamW").strip()]
    bts = str(form.get("ab_blocks_to_swap") or "0").strip()
    if bts and bts != "0":
        argv += ["--blocks_to_swap", bts]
    if form.get("ab_auto_swap"):
        # OOM at the base swap → auto-escalate to the minimal blocks_to_swap that fits.
        argv += ["--max-swap", "26"]
    if form.get("ab_compile"):
        argv += ["--compile"]
    if form.get("ab_auto_budget"):
        # auto-search the activation budget — ab_budget is the LOWEST to try.
        argv += ["--auto-budget", "--min-budget", str(form.get("ab_budget") or "0.1").strip()]
    else:
        bud = str(form.get("ab_budget") or "1.0").strip()
        try:
            if float(bud) < 1.0:  # the activation lever base anima_lora uses (needs compile)
                argv += ["--activation_memory_budget", bud]
        except ValueError:
            pass
    dit = (form.get("ab_dit") or form.get("pretrained_model_name_or_path") or form.get("dit") or "").strip()
    if dit:
        argv += ["--dit", dit]
    return argv, None


def bench_autobatch(form: dict) -> dict:
    """Submit a daemon ``bench-autobatch`` command job — auto-find the max feasible
    batch per resolution for the given network / optimizer / grad-ckpt. Streams to
    the cmd log like a run; the frontier table prints at the end. Daemon-queued so
    it doesn't collide with a training job."""
    argv, err = _autobatch_argv(form)
    if err:
        return {"error": err}
    from scripts.daemon import client as _dc

    try:
        cl = _dc.ensure_daemon(expected_root=str(ROOT))
        resp = cl.submit_command(label="bench-autobatch", argv=argv)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"daemon submit failed: {exc}"}
    _STATE["live_cmd"] = True
    _STATE["run_name"] = "bench-autobatch"
    _start_progress_stream(0.5)
    return {"ok": True, "job_id": resp.get("job_id"), "argv": argv}


def launch(form: dict) -> dict:
    proc = _STATE.get("proc")
    if proc is not None and proc.poll() is None:
        return {"ok": False, "error": "A direct run is already in progress."}

    # Auto-preprocess: rewrite form['dataset_config'] to the fresh-cache config and
    # build the preprocess command-job spec, to be chained → train via the daemon.
    prep = None
    if form.get("auto_preprocess"):
        prep = _prepare_auto_preprocess(form)
        if prep.get("error"):
            return {"ok": False, "error": prep["error"]}
    elif form.get("subsets") and not (form.get("dataset_config") or "").strip():
        # Auto-preprocess OFF + a dataset defined → treat the subsets as already
        # resized + cached and build the dataset config straight from them.
        pc = _build_precached_config(form)
        if pc:
            form["dataset_config"] = pc

    if form.get("dry_run"):
        cmd = " ".join(build_command(form))
        if prep:
            pj = "tasks.py " + " ".join(prep["argv"][1:])
            envs = " ".join(f"{k}={v}" for k, v in prep["extra_env"].items())
            cmd = (
                f"# 1) preprocess job (daemon): {pj}\n#    env: {envs}\n"
                f"#    dataset → {prep['dataset_config']}\n# 2) then chains → train:\n{cmd}"
            )
        return {"ok": True, "dry_run": True, "command": cmd}

    method, preset, extra = _method_preset_extra(form)
    mon = _monitor_url(form)
    cmd_str = " ".join(build_command(form))
    fallback_note = None

    # Robust path (default): submit to the local training daemon — detached, so
    # training SURVIVES the GUI closing; it also queues + captures logs. Same
    # path as `make lora --queue`. Falls back to a direct Popen if unreachable.
    if form.get("daemon", True):
        try:
            from scripts.daemon import client as _dc

            # Pin the daemon to THIS checkout. Without expected_root the GUI would
            # attach to whatever daemon answers the shared ~/.anima pidfile —
            # including one from a different/older anima_lora checkout, which then
            # runs our job with ITS code (e.g. "Unknown command: preprocess-manifest")
            # and it dies instantly. expected_root makes the client shut down an
            # idle foreign-root daemon and spawn one rooted here (it raises instead
            # if that daemon still has live jobs, rather than stealing them).
            cl = _dc.ensure_daemon(expected_root=str(ROOT))
            did_prep = bool(prep) and not _caches_ready(prep)
            if did_prep:
                # preprocess command-job → auto-chains the train job on success
                # (manager._finalize). One Start click; both phases survive close.
                resp = cl.submit_command(
                    label=prep["target"],
                    argv=prep["argv"],
                    extra_env=prep["extra_env"],
                    chain_train={"method": method, "preset": preset, "extra": extra},
                )
                note = "auto-preprocess → train chain submitted"
            else:
                # No preprocess needed: either auto-preprocess is off, or the cache
                # is already complete for this exact spec (marker matched) → train
                # straight away, skipping the 1–2 min re-scan.
                resp = cl.submit(method=method, preset=preset, extra=extra)
                note = (
                    "caches already complete → preprocess skipped, training directly"
                    if prep
                    else None
                )
            _STATE.update(
                proc=None,
                cmd=None,
                started_at=None,
                monitor_url=mon,
                daemon_job=resp.get("job_id"),
                daemon_base=getattr(cl, "base", None),
            )
            # Real-time tqdm bar in the GUI's cmd window (LoRA_Easy-style \r live
            # gauge), unless the user turned it off. Default on.
            _STATE["live_cmd"] = form.get("live_cmd_progress", True) is not False
            _STATE["run_name"] = (form.get("output_name") or "").strip() or "training"
            if _STATE["live_cmd"]:
                _start_progress_stream(form.get("cmd_progress_interval") or 0.5)
            else:
                _stop_progress_stream()
            return {
                "ok": True,
                "daemon": True,
                "job_id": resp.get("job_id"),
                "daemon_base": getattr(cl, "base", None),
                "monitor_url": mon,
                "preprocess": did_prep,
                "command": cmd_str,
                "note": note,
            }
        except Exception as exc:  # noqa: BLE001 — fall back to a direct spawn
            fallback_note = f"daemon unavailable ({exc}); ran directly instead"

    # Direct-spawn fallback can't express the preprocess→train chain (no job queue).
    if prep:
        return {
            "ok": False,
            "error": "Auto-preprocess needs the daemon (it chains "
            "preprocess → train). Enable 'Run via daemon' or preprocess manually.",
        }

    cmd = build_command(form)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"failed to spawn: {exc}"}
    _STATE.update(
        proc=proc, cmd=cmd, started_at=time.time(), monitor_url=mon, daemon_job=None
    )
    return {
        "ok": True,
        "command": cmd_str,
        "pid": proc.pid,
        "monitor_url": mon,
        "note": fallback_note,
    }


def _pct(done, total):
    """done/total → an int 0–100, or None when either is missing/zero."""
    try:
        if total and float(total) > 0:
            return max(0, min(100, round(float(done) * 100.0 / float(total))))
    except (TypeError, ValueError):
        pass
    return None


def _log_tail(stdout_path, n: int = 12):
    """Last ``n`` human-readable lines of a job's ``stdout.log``.

    tqdm rewrites its bar in place with carriage returns, so a raw read yields
    one giant CR-laden line; split on ``\\r`` too and keep the final segment of
    each so the tail reads like the live console. Best-effort → ``[]`` on error.
    """
    if not stdout_path:
        return []
    try:
        data = Path(stdout_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out = []
    for raw in data.splitlines():
        seg = raw.split("\r")[-1].rstrip()  # last CR segment = current bar state
        if seg:
            out.append(seg)
    return out[-n:]


def _report_and_tail(job, n: int = 40):
    """Tail a failed daemon job's ``stdout.log`` AND print it once to the webgui's
    own console (stderr), so the terminal running the GUI shows WHY a job died.

    Daemon jobs run windowless (pythonw, no console), so their only record is the
    per-job log file — the GUI process is the one with a terminal. Deduped per
    (job_id, state) so the 2-second status poll doesn't spam the same traceback.
    Returns the tail (also surfaced to the browser as ``error_log``).
    """
    import sys as _sys

    tail = _log_tail(job.get("stdout_path"), n=n)
    jid = job.get("id") or job.get("job_id")
    state = job.get("state")
    seen = _STATE.setdefault("_reported_errs", set())
    key = (jid, state)
    if jid and key not in seen:
        seen.add(key)
        err = job.get("error")
        body = "\n".join(tail) or "(no stdout captured)"
        _sys.stderr.write(
            f"\n{'=' * 72}\n[webgui] daemon job {jid} {str(state).upper()}"
            f"{(' — ' + err) if err else ''}\n  log: {job.get('stdout_path')}\n"
            f"  --- last {len(tail)} log line(s) ---\n{body}\n{'=' * 72}\n"
        )
        _sys.stderr.flush()
    return tail


def _extract_tqdm_line(stdout_path):
    """The latest tqdm progress line — of ANY kind (training, sampling, loading) —
    in a job's stdout.log, for the throttled new-line fallback used when the live
    tail is off. Sampling is no longer filtered out, so a sample pass is visible.

    The daemon runs train.py detached (stdout → file); tqdm rewrites its bar in
    place with \\r, so scan from the end for the most recent line carrying it/s or
    s/it + an N/total fraction. ``None`` if there isn't one yet.
    """
    if not stdout_path:
        return None
    try:
        data = Path(stdout_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = [s.strip() for s in data.replace("\r", "\n").split("\n")]
    for seg in reversed(lines):
        if seg and ("it/s" in seg or "s/it" in seg) and "/" in seg:
            return seg
    return None


def _report_progress(job):
    """Throttled (~3s) NEW-LINE progress mirror — the fallback when the live \\r
    streamer is OFF. Skipped when ``_STATE['live_cmd']`` (the streamer owns the
    console then, so we don't double-print)."""
    if _STATE.get("live_cmd"):
        return
    line = _extract_tqdm_line(job.get("stdout_path"))
    if not line:
        return
    st = _STATE.setdefault("_prog", {"t": 0.0, "line": None})
    now = time.time()
    if line == st.get("line") or (now - st.get("t", 0.0)) < 3.0:
        return
    st["t"], st["line"] = now, line
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


_progress_gen = 0


def _stop_progress_stream():
    """Bump the generation so any running live-progress thread exits promptly."""
    global _progress_gen
    _progress_gen += 1


def _start_progress_stream(interval=0.5):
    """Tail the ACTIVE daemon job's stdout.log to the webgui console in REAL TIME —
    a full, auto-scrolling training log (loading, training, AND sampling / decode
    lines, nothing filtered), the way a normal trainer's console reads. Replaces the
    old single-line \\r gauge so a sample pass is visible instead of hidden. Prints a
    header with the run's output_name so logs are identifiable. Runs in a daemon
    thread that follows whichever job is active; a new run (or stop) bumps the
    generation → the old tailer exits; it also self-exits ~12s after nothing runs.
    """
    global _progress_gen
    _progress_gen += 1
    gen = _progress_gen
    interval = max(0.1, min(float(interval or 0.5), 5.0))

    def _loop():
        from scripts.daemon import client as _dc

        pos = {}  # stdout_path → byte offset already mirrored
        header_for = None  # active job id we've printed a header for
        idle, err_last = 0.0, None
        while gen == _progress_gen:
            try:
                cl = _dc.DaemonClient()
                active = (cl.health() or {}).get("active_job")
                err_last = None  # daemon reachable → clear any prior error note
                if not active:
                    idle += interval
                    if idle > 12.0:  # nothing running for a while → run finished
                        break
                    time.sleep(interval)
                    continue
                idle = 0.0
                path = (cl.get(active) or {}).get("stdout_path")
                if not path:
                    time.sleep(interval)
                    continue
                if active != header_for:
                    # New job: print a header + start tailing from the CURRENT end so
                    # a reconnect doesn't replay the whole log (a fresh launch's log is
                    # near-empty, so this still catches it from the top).
                    header_for = active
                    name = _STATE.get("run_name") or "training"
                    sys.stdout.write(f"\n=== Real-time training logs — {name} ===\n")
                    sys.stdout.flush()
                    try:
                        pos[path] = os.path.getsize(path)
                    except OSError:
                        pos[path] = 0
                try:
                    with open(path, "rb") as f:
                        f.seek(pos.get(path, 0))
                        chunk = f.read()
                        pos[path] = f.tell()
                except OSError:
                    time.sleep(interval)
                    continue
                if chunk:
                    if gen != _progress_gen:  # superseded mid-iteration → stop writing
                        break
                    # Write raw (keep \r) so a tqdm bar updates IN PLACE (one climbing
                    # line) while real log lines (\n) scroll — the native trainer
                    # console look: loading/sampling bars don't pile up line-by-line.
                    sys.stdout.write(chunk.decode("utf-8", "replace"))
                    sys.stdout.flush()
            except Exception as exc:  # noqa: BLE001 — best-effort console mirror
                # Never crash the GUI over a transient daemon hiccup, but surface it
                # ONCE (deduped) so a frozen tail is debuggable instead of silent.
                msg = f"{type(exc).__name__}: {exc}"
                if msg != err_last:
                    err_last = msg
                    try:
                        sys.stderr.write(f"\n[log stream] {msg}\n")
                        sys.stderr.flush()
                    except Exception:  # noqa: BLE001
                        pass
            time.sleep(interval)
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_loop, daemon=True).start()


def _train_phase(cl, train_id, out, job=None):
    """Fill ``out`` from a daemon train job's state (running → embed monitor)."""
    try:
        tj = job if job is not None else (cl.get(train_id) or {})
    except Exception:  # noqa: BLE001 — daemon best-effort
        tj = {}
    st = tj.get("state")
    out["training"] = {"job_id": train_id, "state": st}
    if st == "running":
        out["phase"] = "training"
        out["training_started"] = True
        _report_progress(tj)  # mirror progress to the GUI's cmd window
    elif st == "queued":
        out["phase"] = "train_queued"
    elif st == "done":
        out["phase"] = "done"
    elif st == "stopped":
        out["phase"] = "stopped"
    elif st == "error":
        out["phase"] = "error"
        out["error_log"] = _report_and_tail(tj)
    else:
        out["phase"] = "training"  # unknown, but a train job exists
    return out


def _daemon_phase(tracked_job_id):
    """Resolve the live phase of a daemon run (preprocess → chained train).

    ``tracked_job_id`` is whatever ``launch()`` recorded in ``_STATE`` — the
    preprocess command job when auto-preprocess is on, else the train job
    itself. Returns the phase model the frontend renders: a preprocess progress
    bar that auto-hands-off to the embedded monitor the moment training starts.
    Best-effort: daemon down / unknown job → an inert ``{phase: None}`` so the
    legacy poll fields still drive the UI.
    """
    out = {
        "phase": None,
        "preprocess": None,
        "training": None,
        "training_started": False,
    }
    try:
        from scripts.daemon import client as _dc

        cl = _dc.DaemonClient()
        if cl.health() is None:
            return out
        job = cl.get(tracked_job_id) or {}
    except Exception:  # noqa: BLE001 — daemon is optional infra here
        return out
    # Bail only on the daemon's 404 shape ({"error":"no such job"} — no state).
    # A REAL job that FAILED also carries an `error` field, and we very much want
    # to surface THAT (don't conflate "job not found" with "job errored").
    if not job or not job.get("state"):
        return out
    if job.get("kind") == "command":
        state = job.get("state")
        latest = job.get("latest") or {}
        out["preprocess"] = {
            "state": state,
            "phase_label": latest.get("phase"),
            "done": latest.get("done"),
            "total": latest.get("total"),
            "percent": _pct(latest.get("done"), latest.get("total")),
            "log_tail": _log_tail(job.get("stdout_path"))
            if state in ("queued", "running")
            else [],
        }
        if state in ("queued", "running"):
            out["phase"] = "preprocess"
            return out
        if state == "stopped":
            out["phase"] = "stopped"
            return out
        if state == "error":
            out["phase"] = "error"
            out["error_log"] = _report_and_tail(job)
            return out
        # preprocess done → follow the chained training job, if one was spawned
        chained = job.get("chained_job_id")
        if chained:
            return _train_phase(cl, chained, out)
        out["phase"] = "done"
        return out
    # the tracked job is itself a train job (no preprocess step)
    return _train_phase(cl, tracked_job_id, out, job=job)


def status() -> dict:
    proc = _STATE.get("proc")
    running = proc is not None and proc.poll() is None
    out = {
        "running": running,
        "pid": proc.pid if proc else None,
        "returncode": (proc.poll() if proc else None) if not running else None,
        "command": " ".join(_STATE["cmd"]) if _STATE.get("cmd") else None,
        "monitor_url": _STATE.get("monitor_url"),
        "elapsed": (time.time() - _STATE["started_at"])
        if _STATE.get("started_at") and running
        else None,
        "daemon_job": _STATE.get("daemon_job"),
        "daemon_base": _STATE.get("daemon_base"),
    }
    # Enrich with the daemon phase model (preprocess bar → training hand-off)
    # whenever a daemon job is tracked. Guarded so a daemon hiccup can never
    # break the status poll the whole UI depends on.
    djob = _STATE.get("daemon_job")
    if djob:
        try:
            out.update(_daemon_phase(djob))
        except Exception:  # noqa: BLE001
            pass
    return out


def stop() -> dict:
    _stop_progress_stream()  # halt the live cmd bar
    # Direct-Popen path (daemon disabled): terminate the spawned process.
    proc = _STATE.get("proc")
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "stopped": "direct"}

    # Daemon path (the default): the job runs DETACHED in the daemon, so
    # _STATE['proc'] is None — terminating it does nothing. Tell the daemon to
    # stop the run instead. We stop the tracked job, its chained follow-on, AND
    # the currently active job, so a running preprocess/train is killed (tree +
    # GPU freed) and a still-queued follow-on can't sneak through. (Stopping a
    # preprocess before it finishes also blocks the chain: the manager only
    # enqueues the train job when the command job reaches `done`.)
    djob = _STATE.get("daemon_job")
    if djob:
        try:
            from scripts.daemon import client as _dc

            cl = _dc.DaemonClient()
            if cl.health() is None:
                return {"ok": False, "error": "training daemon not reachable"}
            ids = [djob]
            job = cl.get(djob) or {}
            if job.get("chained_job_id"):
                ids.append(job["chained_job_id"])
            active = (cl.health() or {}).get("active_job")
            if active:
                ids.append(active)
            stopped = []
            for jid in dict.fromkeys(i for i in ids if i):
                try:
                    r = cl.stop(jid) or {}
                    stopped.append({"job": jid, "state": r.get("state")})
                except Exception:  # noqa: BLE001 — already terminal / gone
                    pass
            return {"ok": True, "stopped": "daemon", "jobs": stopped}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"daemon stop failed: {exc}"}

    return {"ok": False, "error": "no run in progress"}


# --------------------------------------------------------------------------- #
# Queue + saved-config store (persisted JSON, LoRA_Easy-style)
# --------------------------------------------------------------------------- #
STORE_DIR = Path(__file__).resolve().parent / "store"
QUEUE_FILE = STORE_DIR / "queue.json"
CONFIG_DIR = STORE_DIR / "configs"
DATASET_DIR = STORE_DIR / "datasets"


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _safe_name(name: str) -> str:
    out = "".join(c for c in (name or "") if c.isalnum() or c in "-_ ").strip()
    return out or "config"


def queue_list() -> list:
    return _read_json(QUEUE_FILE, [])


def queue_add(name: str, form: dict) -> dict:
    q = queue_list()
    next_id = max((int(i.get("id", 0)) for i in q), default=0) + 1
    q.append({"id": next_id, "name": (name or f"job {len(q) + 1}"), "form": form})
    _write_json(QUEUE_FILE, q)
    return {"ok": True, "queue": q}


def queue_remove(item_id) -> dict:
    q = [i for i in queue_list() if str(i.get("id")) != str(item_id)]
    _write_json(QUEUE_FILE, q)
    return {"ok": True, "queue": q}


def queue_reorder(order: list) -> dict:
    q = queue_list()
    by_id = {str(i["id"]): i for i in q}
    seen = {str(x) for x in order}
    new = [by_id[str(i)] for i in order if str(i) in by_id]
    new += [i for i in q if str(i["id"]) not in seen]  # keep any not listed
    _write_json(QUEUE_FILE, new)
    return {"ok": True, "queue": new}


def queue_clear() -> dict:
    _write_json(QUEUE_FILE, [])
    return {"ok": True, "queue": []}


def queue_run() -> dict:
    """Submit every queued job to the training daemon in order (it then runs
    them sequentially). The queue is left intact — clear it if you want."""
    results = []
    for item in queue_list():
        form = dict(item.get("form") or {})
        form["daemon"] = True
        form.pop("dry_run", None)
        r = launch(form)
        results.append(
            {
                "name": item.get("name"),
                "ok": bool(r.get("ok")),
                "job_id": r.get("job_id"),
                "error": r.get("error"),
            }
        )
    return {"ok": True, "submitted": results}


def config_list() -> list:
    return (
        sorted(p.stem for p in CONFIG_DIR.glob("*.json")) if CONFIG_DIR.is_dir() else []
    )


def config_save(name: str, form: dict) -> dict:
    n = _safe_name(name)
    _write_json(CONFIG_DIR / f"{n}.json", form)
    return {"ok": True, "name": n, "configs": config_list()}


def config_load(name: str) -> dict:
    p = CONFIG_DIR / f"{_safe_name(name)}.json"
    if not p.exists():
        return {"ok": False, "error": "not found"}
    return {"ok": True, "form": _read_json(p, {})}


def config_delete(name: str) -> dict:
    p = CONFIG_DIR / f"{_safe_name(name)}.json"
    try:
        if p.exists():
            p.unlink()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "configs": config_list()}


# Valid anima_lora subset keys the builder may emit. These all pass
# `library.config.dataset_keys.lint_dataset_sections` (the training validator).
# NOTE: `caption_tag_dropout_rate` is intentionally EXCLUDED — it is a real subset
# key but HARD-CRASHES under the default `use_text_cache=true` (the TE cacheability
# assert), so per-tag dropout is a preprocess knob (`--caption_tag_dropout_rate` on
# cache_text_embeddings) instead. Likewise `shuffle_caption` is NOT a dataset key
# and NOT a CLI flag in anima — caption shuffling is `--caption_shuffle_variants`
# at TE-cache time (the Preprocess panel), not a train-time toggle.
_SUBSET_KEYS = {
    "image_dir": str,
    "cache_dir": str,
    "mask_dir": str,
    "num_repeats": int,
    "keep_tokens": int,
    "caption_extension": str,
    "recursive": bool,
    "caption_dropout_rate": float,
    "caption_dropout_every_n_epochs": int,
    "flip_aug": bool,
    "color_aug": bool,
    "random_crop": bool,
    "caption_prefix": str,
    "caption_suffix": str,
}


def _flatten_kv(v) -> str:
    """A dict {k:val} / list ['k=v'] / 'k=v k=v' string → a 'k=v k=v' string."""

    def _val(x):
        # List/tuple values must render WITHOUT internal spaces — the result is
        # space-split downstream (optimizer_args / network_args), so "betas=[0.9, 0.999]"
        # would shatter into "betas=[0.9," + "0.999]". Compact to "betas=[0.9,0.999]".
        if isinstance(x, (list, tuple)):
            return "[" + ",".join(str(e) for e in x) + "]"
        return str(x)

    if isinstance(v, dict):
        return " ".join(f"{k}={_val(vv)}" for k, vv in v.items())
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v or "").strip()


def _snap_tier(res) -> int | None:
    """Map a kohya resolution (int / [W,H] / 'W,H') to the nearest anima tier edge."""
    import re as _re

    edges = list_target_res_tiers()
    r = None
    if isinstance(res, (list, tuple)) and res:
        r = max(int(x) for x in res)
    elif isinstance(res, (int, float)):
        r = int(res)
    elif isinstance(res, str):
        nums = [int(x) for x in _re.findall(r"\d+", res)]
        r = max(nums) if nums else None
    return min(edges, key=lambda e: abs(e - r)) if r else None


# kohya / LoRA_Easy keys with NO anima equivalent — dropped on import (anima uses
# constant-token buckets + preprocess-time caption shuffle, not these). Anima-VALID
# args (no_half_vae / prior_loss_weight / min_snr_gamma / reg_data_dir / …) are NOT
# here — the comprehensive pass-through in import_config forwards them as auto-args.
_IMPORT_DROP = {
    "train_mode",  # LoRA_Easy meta key
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "bucket_no_upscale",
    "multires_training",
    "skip_image_resolution",  # → "skip upscaling" auto / per-tier ms_skip
    "shuffle_caption",  # → preprocess caption_shuffle_variants
    "caption_tag_dropout_rate",  # → preprocess tag-drop (train-time crashes the TE cache)
    "sdxl",
    "v2",
    "v_parameterization",
    "clip_skip",
    "xformers",  # → attn_mode
    "split_attn",  # not an anima knob
    "edm2_loss_weighting",
    "save_toml",
    "save_toml_location",
    "cache_latents",  # anima caches by default (use_vae_cache)
    "cache_latents_to_disk",
    "name",
}


def import_config(path: str) -> dict:
    """Parse an anima dataset .toml OR a LoRA_Easy / kohya config and map it onto
    the GUI form dict. Returns ``{ok, form, subsets, notes}``.

    Detects: LoRA_Easy sectioned (``*_args.args``/``subsets``/``train_mode``),
    kohya sectioned (``*_arguments``), kohya GUI ``.json`` (flat), or an
    anima/kohya dataset blueprint (``[[datasets]]``). Incompatible keys
    (``_IMPORT_DROP`` — enable_bucket/skip_image_resolution/shuffle_caption/…) are
    stripped; ``resolution`` → nearest ``target_res`` tier(s). Only fields it
    actually found are returned, so ``setForm`` merges onto current defaults.
    """
    p = Path((path or "").strip().strip('"'))
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {p}"}
    text = p.read_text(encoding="utf-8")
    notes: list[str] = []

    flat: dict = {}
    subsets: list = []
    datasets: list = []
    if p.suffix.lower() == ".json":
        import json as _json

        flat = dict(_json.loads(text))  # kohya GUI flat json
    else:
        import tomllib

        raw = tomllib.loads(text)
        # LoRA_Easy sectioned is defined by the `.args` / `.dataset_args` nesting
        # (NOT a bare `train_mode`/`subsets` key — those also appear in flat configs).
        le_sectioned = any(
            isinstance(v, dict) and ("args" in v or "dataset_args" in v)
            for v in raw.values()
        )
        kohya_sectioned = any(k.endswith("_arguments") for k in raw)
        subsets = list(raw.get("subsets") or [])
        datasets = list(raw.get("datasets") or [])
        if le_sectioned:
            for val in raw.values():
                if not isinstance(val, dict):
                    continue
                for sub in ("args", "dataset_args"):
                    if isinstance(val.get(sub), dict):
                        flat.update(val[sub])
        elif kohya_sectioned:
            for val in raw.values():
                if isinstance(val, dict):
                    flat.update(val)
        else:
            # flat / hand-written (config_anima.toml) — keep scalars AND list keys
            # (network_args / optimizer_args are lists here); drop only the blueprint
            # sections handled separately.
            flat = {
                k: v
                for k, v in raw.items()
                if k not in ("datasets", "general", "subsets")
            }

    # Dataset blueprint (anima/kohya): pull subsets from the [[datasets]] blocks,
    # annotating each with its block's resolution / batch_size / skip so a multi-
    # block (multi-resolution) kohya dataset maps to per-subset tiers + batch (the
    # GUI multi-scale per-block model) instead of flattening to one tier.
    res_for_tier = flat.get("resolution")
    _block_tiers: list[int] = []
    if datasets and not subsets:
        for ds in datasets:
            if not isinstance(ds, dict):
                continue
            res_for_tier = res_for_tier or ds.get("resolution")
            d_tier = (
                _snap_tier(ds.get("resolution"))
                if ds.get("resolution") is not None
                else None
            )
            d_bs = ds.get("batch_size")
            if d_tier and d_tier not in _block_tiers:
                _block_tiers.append(d_tier)
            for sub in ds.get("subsets") or []:
                if isinstance(sub, dict):
                    sub = dict(sub)
                    if d_tier:
                        sub["_tier"] = d_tier
                    if d_bs is not None:
                        sub["_batch"] = d_bs
                    subsets.append(sub)

    form: dict = {"method": "lora"}

    def _put(key, *src_keys, cast=str):
        for sk in src_keys:
            if sk in flat and flat[sk] not in (None, ""):
                try:
                    form[key] = cast(flat[sk]) if cast is not str else str(flat[sk])
                except (TypeError, ValueError):
                    form[key] = str(flat[sk])
                return

    _put("dit_path", "pretrained_model_name_or_path")
    _put("te_path", "qwen3")
    _put("vae_path", "vae")
    _put("network_dim", "network_dim")
    _put("network_alpha", "network_alpha")
    _put("optimizer_type", "optimizer_type", "optimizer")
    _put("learning_rate", "learning_rate", "lr", "unet_lr")
    _put("max_train_epochs", "max_train_epochs")
    _put("output_name", "output_name")
    _put("output_dir", "output_dir")
    _put("resume", "resume")
    _put("sample_prompts", "sample_prompts")
    _put("seed", "seed")

    # network module / algo / preset / extra net args
    nm = str(flat.get("network_module") or "").strip()
    na = flat.get("network_args")
    na_dict = dict(na) if isinstance(na, dict) else {}
    if isinstance(na, (list, tuple)):
        for item in na:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                # list-valued kwargs (e.g. exclude_patterns=['a', 'b']) must lose
                # internal spaces — network_args are space-split downstream, so a
                # space after the comma would shatter the list into broken tokens.
                if v.lstrip().startswith("[") and " " in v:
                    v = v.replace(", ", ",").replace(" ,", ",")
                na_dict[k] = v
    if nm:
        if "lycoris" in nm:
            # Stock lycoris.kohya crashes on Anima's [None] TE slot — always route
            # to the Anima-safe bridge.
            form["network_module"] = "networks.lycoris_anima"
            if nm != "networks.lycoris_anima":
                notes.append(
                    f"network_module {nm!r} → networks.lycoris_anima (Anima LyCORIS bridge)."
                )
        elif nm.startswith("networks."):
            form["network_module"] = nm
        else:
            form["network_module"] = "networks.lora_anima"
    if "algo" in na_dict:
        form["algo"] = str(na_dict.pop("algo"))
        # LoRA_Easy/kohya lycoris configs often omit network_module — if an algo
        # is present, route to the Anima-safe lycoris bridge so the algo applies.
        if "lycoris" not in (form.get("network_module") or ""):
            form["network_module"] = "networks.lycoris_anima"
            notes.append("network_module → networks.lycoris_anima (algo present).")
    if "preset" in na_dict:
        pv = str(na_dict.pop("preset"))
        # Stock LyCORIS presets target diffusers class names absent from the Anima
        # DiT — remap to the Anima preset that actually wraps its blocks.
        _stock = {
            "full",
            "full-lin",
            "attn-mlp",
            "attn-only",
            "unet-only",
            "unet-transformer-only",
            "unet-convblock-only",
            "ia3",
        }
        if pv in _stock or pv not in list_lycoris_presets():
            form["lycoris_preset"] = "anima-attn-mlp"
            notes.append(
                f"preset {pv!r} → anima-attn-mlp (stock presets don't wrap the Anima DiT)."
            )
        else:
            form["lycoris_preset"] = pv
    leftover_na = _flatten_kv(na_dict)
    if leftover_na:
        form["network_args_extra"] = leftover_na

    # optimizer / scheduler args (dict/list/str → 'k=v' string)
    if flat.get("optimizer_args"):
        form["optimizer_args"] = _flatten_kv(flat["optimizer_args"])
    if str(flat.get("lr_scheduler_type") or "").strip():
        form["lr_scheduler_type"] = str(flat["lr_scheduler_type"])
    elif str(flat.get("lr_scheduler") or "").strip():
        form["lr_scheduler_type"] = str(flat["lr_scheduler"])
    if flat.get("lr_scheduler_args"):
        form["lr_scheduler_args"] = _flatten_kv(flat["lr_scheduler_args"])
    if str(flat.get("lr_warmup_steps") or "").strip():
        try:
            if float(flat["lr_warmup_steps"]) >= 1:
                form["lr_warmup_steps"] = str(flat["lr_warmup_steps"])
            else:
                notes.append("lr_warmup_steps was a ratio (<1) — left blank.")
        except (TypeError, ValueError):
            pass
    if "warmup_ratio" in flat:
        notes.append(
            "warmup_ratio can't convert to steps without total steps — set LR warmup manually."
        )

    # resolution → target_res tier(s). Multi-block kohya dataset (one [[datasets]]
    # block per resolution) → multi-scale across each block's tier; the per-tier
    # "skip upscaling" auto-default (next-lower tier) reproduces skip_image_resolution.
    if len(_block_tiers) >= 2:
        form["target_res"] = [str(t) for t in sorted(set(_block_tiers))]
        form["multiscale"] = True
        notes.append(
            f"{len(datasets)} dataset blocks → multi-scale tiers "
            f"{sorted(set(_block_tiers))} (per-block batch/repeat kept; "
            "skip_image_resolution → auto 'skip upscaling')."
        )
    else:
        tier = _snap_tier(res_for_tier) if res_for_tier is not None else None
        if tier:
            form["target_res"] = [str(tier)]
            notes.append(
                f"resolution {res_for_tier} → target_res tier {tier} (anima constant-token)."
            )

    # Flow-matching / timestep settings → auto-arg overrides (the form's adv[]),
    # which setForm checks on + fills in. kohya's discrete min/max_timestep (0~1000)
    # become anima's continuous t_min/t_max SIGMA (÷1000): min_timestep=0 → t_min=0.0,
    # max_timestep=1000 → t_max=1.0. The rest map name-for-name.
    adv: list = []

    def _adv(flag: str, val) -> None:
        if val not in (None, ""):
            adv.append({"flag": flag, "value": str(val), "is_bool": False})

    if "min_timestep" in flat:
        try:
            _adv("--t_min", round(float(flat["min_timestep"]) / 1000.0, 6))
            notes.append("min_timestep → t_min (÷1000, anima uses sigma 0~1).")
        except (TypeError, ValueError):
            pass
    if "max_timestep" in flat:
        try:
            _adv("--t_max", round(float(flat["max_timestep"]) / 1000.0, 6))
            notes.append("max_timestep → t_max (÷1000, anima uses sigma 0~1).")
        except (TypeError, ValueError):
            pass
    for src, flag in (
        ("timestep_sample_method", "--timestep_sampling"),
        ("timestep_sampling", "--timestep_sampling"),
        ("sigmoid_scale", "--sigmoid_scale"),
        ("sigmoid_bias", "--sigmoid_bias"),
        ("discrete_flow_shift", "--discrete_flow_shift"),
        ("weighting_scheme", "--weighting_scheme"),
        ("logit_mean", "--logit_mean"),
        ("logit_std", "--logit_std"),
        ("max_token_length", "--qwen3_max_token_length"),
    ):
        if src in flat:
            _adv(flag, flat[src])

    # Comprehensive pass-through: every other flat key that is a REAL anima
    # argparse arg (curated keys are excluded from list_arg_groups; special-mapped
    # timestep keys handled above) rides adv[] verbatim — so the whole training
    # config imports, not just the curated few. Bools emit the flag only when
    # truthy; list values are space-joined.
    _special_src = {
        "min_timestep", "max_timestep", "timestep_sample_method", "timestep_sampling",
        "sigmoid_scale", "sigmoid_bias", "discrete_flow_shift", "weighting_scheme",
        "logit_mean", "logit_std", "max_token_length",
    }
    _seen_dests = {a["flag"].lstrip("-") for a in adv}
    _valid = {a["dest"]: a for g in list_arg_groups() for a in g["args"]}
    for k, v in flat.items():
        if k in _IMPORT_DROP or k in _special_src or k in _seen_dests:
            continue
        meta = _valid.get(k)
        if meta is None:  # not an anima arg (or curated → handled above)
            continue
        if meta["is_bool"]:
            truthy = v is True or str(v).strip().lower() in ("true", "1", "yes")
            if meta.get("negatable"):
                # tri-state: preserve an explicit false as "off" so a config that turns
                # a base.toml-true flag back off round-trips (was silently dropped).
                adv.append({"flag": meta["flag"], "is_bool": True, "negatable": True,
                            "tri": "on" if truthy else "off"})
            elif truthy:
                adv.append({"flag": meta["flag"], "is_bool": True, "value": True})
        elif v not in (None, "", []):
            val = (
                " ".join(str(x) for x in v)
                if isinstance(v, (list, tuple))
                else str(v)
            )
            adv.append(
                {"flag": meta["flag"], "is_bool": False, "value": val, "nargs": meta.get("nargs")}
            )
    if adv:
        form["adv"] = adv

    # subsets → the manual builder's shape (only valid anima keys)
    out_subs = []
    for s in subsets:
        if not isinstance(s, dict) or not s.get("image_dir"):
            continue
        entry = {
            "image_dir": s.get("image_dir"),
            "num_repeats": s.get("num_repeats", 1),
            "keep_tokens": s.get("keep_tokens", 0),
            "caption_extension": s.get("caption_extension", ".txt"),
            "caption_dropout_rate": s.get("caption_dropout_rate", 0),
            "flip_aug": bool(s.get("flip_aug")),
            "random_crop": bool(s.get("random_crop")),
            "random_crop_padding_percent": s.get("random_crop_padding_percent", 0.05),
            "recursive": bool(s.get("recursive")),
        }
        # per-block tier + batch (multi-resolution kohya dataset → per-subset)
        if s.get("_tier"):
            entry["tiers"] = [s["_tier"]]
        if s.get("_batch") is not None:
            entry["batch_size"] = s["_batch"]
        out_subs.append(entry)

    dropped = sorted(k for k in _IMPORT_DROP if k in flat)
    if dropped:
        notes.append("dropped anima-incompatible keys: " + ", ".join(dropped))
    if not form.get("te_path"):
        notes.append("no text-encoder (qwen3) path in source — set it in Model files.")

    return {"ok": True, "form": form, "subsets": out_subs, "notes": notes}


def build_dataset_toml(data: dict) -> dict:
    """Write an anima_lora-compatible dataset config TOML from the GUI builder
    (one or more image subsets) and return its path — set it as dataset_config.

    `mask_dir` (per subset) wires SAM3/MIT masks into masked_loss; the GUI sets it
    when masking is enabled (default `post_image_dataset/masks`). Caption shuffling
    is NOT emitted here — it's `--caption_shuffle_variants` at preprocess time."""
    import toml as _toml

    name = _safe_name(data.get("name") or "dataset")
    # A top-level mask_dir (set by the Masking panel) applies to every subset that
    # doesn't override it — so enabling masking wires masked_loss without the user
    # editing each subset.
    global_mask_dir = (data.get("mask_dir") or "").strip()
    ds = {"batch_size": int(data.get("batch_size") or 1), "subsets": []}
    for s in data.get("subsets") or []:
        sub = {}
        for k, caster in _SUBSET_KEYS.items():
            v = s.get(k)
            if v in (None, ""):
                continue
            try:
                sub[k] = caster(v) if caster is not bool else bool(v)
            except (TypeError, ValueError):
                continue
        if sub.get("image_dir"):
            if global_mask_dir and not sub.get("mask_dir"):
                sub["mask_dir"] = global_mask_dir
            ds["subsets"].append(sub)
    if not ds["subsets"]:
        return {"ok": False, "error": "add at least one subset with an image_dir"}
    toml_str = _toml.dumps({"datasets": [ds]})
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASET_DIR / f"{name}.toml"
    path.write_text(toml_str, encoding="utf-8")
    return {"ok": True, "path": str(path), "toml": toml_str}


# --------------------------------------------------------------------------- #
# Server-side folder browser (the GUI "Browse…" picker — local tool, real FS)
# --------------------------------------------------------------------------- #
def browse(path: str | None, exts: str | None = None) -> dict:
    """List subdirectories (and optionally files) of ``path`` for the GUI picker.

    A browser can't open a native file/folder dialog with a real on-disk path, so
    the local server lists the filesystem itself and the client navigates. Empty
    path → Windows drive letters (or ``/`` on POSIX). Returns the resolved dir, its
    parent, each subfolder as ``{name, path}`` in ``dirs``, and — when ``exts`` is
    given (comma-separated, e.g. ``"safetensors,toml"``) — matching files in
    ``files`` so file pickers (model weights, dataset .toml) can select directly.
    """
    import string

    ext_set = (
        {e.strip().lstrip(".").lower() for e in exts.split(",") if e.strip()}
        if exts
        else None
    )
    p = (path or "").strip()
    if not p:
        if os.name == "nt":
            drives = [
                f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")
            ]
            return {
                "ok": True,
                "path": "",
                "parent": None,
                "dirs": [{"name": d, "path": d} for d in drives],
                "files": [],
            }
        p = "/"
    try:
        base = Path(p)
        if not base.is_dir():
            base = base.parent if base.parent.is_dir() else Path.home()
        base = base.resolve()
        entries = list(os.scandir(base))
        subs = sorted(
            (e for e in entries if e.is_dir() and not e.name.startswith(".")),
            key=lambda e: e.name.lower(),
        )
        dirs = [{"name": e.name, "path": str(Path(base, e.name))} for e in subs]
        files = []
        if ext_set is not None:
            fs = sorted(
                (
                    e
                    for e in entries
                    if e.is_file()
                    and Path(e.name).suffix.lstrip(".").lower() in ext_set
                ),
                key=lambda e: e.name.lower(),
            )
            files = [{"name": e.name, "path": str(Path(base, e.name))} for e in fs]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "path": p, "dirs": [], "files": []}
    same = str(base.parent) == str(base)
    parent = ("" if os.name == "nt" else None) if same else str(base.parent)
    return {
        "ok": True,
        "path": str(base),
        "parent": parent,
        "dirs": dirs,
        "files": files,
    }


# --------------------------------------------------------------------------- #
# Sample-prompt editor — write/read the --sample_prompts .txt (one prompt per
# line, anima `<prompt> --w --h --s --l --g --fs --d --n …` token format).
# --------------------------------------------------------------------------- #
SAMPLE_PROMPT_DIR = STORE_DIR / "sample_prompts"


def save_sample_prompts(name: str, text: str) -> dict:
    """Write the editor's serialized lines to a .txt and return its path so the
    form's ``sample_prompts`` field can point at it."""
    SAMPLE_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_PROMPT_DIR / (_safe_name(name or "sample") + ".txt")
    path.write_text((text or "").strip() + "\n", encoding="utf-8")
    return {"ok": True, "path": str(path)}


def load_sample_prompts(path: str) -> dict:
    """Return the raw text of an existing sample-prompts .txt for the editor
    (the client parses the ``--tokens``)."""
    p = Path((path or "").strip().strip('"'))
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {p}"}
    try:
        return {"ok": True, "text": p.read_text(encoding="utf-8")}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


# --------------------------------------------------------------------------- #
# HTTP server
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                self._send(200, HTML_FILE.read_bytes(), "text/html; charset=utf-8")
            except Exception as exc:  # noqa: BLE001
                self._send(500, str(exc).encode(), "text/plain")
        elif path == "/api/options":
            self._json(options())
        elif path == "/api/status":
            self._json(status())
        elif path == "/api/queue/list":
            self._json({"queue": queue_list()})
        elif path == "/api/config/list":
            self._json({"configs": config_list()})
        elif path == "/api/config/load":
            qs = parse_qs(urlparse(self.path).query or "")
            self._json(config_load((qs.get("name") or [""])[0]))
        elif path == "/api/optimizer_args":
            qs = parse_qs(urlparse(self.path).query or "")
            self._json(optimizer_arg_help((qs.get("name") or [""])[0]))
        elif path == "/api/browse":
            qs = parse_qs(urlparse(self.path).query or "")
            self._json(
                browse((qs.get("path") or [""])[0], (qs.get("exts") or [None])[0])
            )
        elif path == "/api/sample_prompts/load":
            qs = parse_qs(urlparse(self.path).query or "")
            self._json(load_sample_prompts((qs.get("path") or [""])[0]))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        if path == "/api/launch":
            self._json(launch(body))
        elif path == "/api/stop":
            self._json(stop())
        elif path == "/api/queue/add":
            self._json(queue_add(body.get("name", ""), body.get("form", {})))
        elif path == "/api/queue/remove":
            self._json(queue_remove(body.get("id")))
        elif path == "/api/queue/reorder":
            self._json(queue_reorder(body.get("order", [])))
        elif path == "/api/queue/clear":
            self._json(queue_clear())
        elif path == "/api/queue/run":
            self._json(queue_run())
        elif path == "/api/config/save":
            self._json(config_save(body.get("name", ""), body.get("form", {})))
        elif path == "/api/config/delete":
            self._json(config_delete(body.get("name", "")))
        elif path == "/api/dataset/build":
            self._json(build_dataset_toml(body))
        elif path == "/api/config/import":
            self._json(import_config(body.get("path", "")))
        elif path == "/api/sample_prompts/save":
            self._json(save_sample_prompts(body.get("name", ""), body.get("text", "")))
        elif path == "/api/bench_autobatch":
            self._json(bench_autobatch(body))
        else:
            self._json({"error": "not found"}, 404)

    def log_message(self, *args):  # silence default logging
        pass


def serve(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    # The requested port may already be taken — most commonly because 7860 is
    # ALSO gradio's default (sd-webui / forge-neo), or a prior GUI instance is
    # still running. Rather than crash with WinError 10013 / address-in-use (the
    # window just vanishes), scan upward for the first free port and use it.
    server = None
    last_err: OSError | None = None
    for p in range(port, port + 20):
        try:
            server = HTTPServer((host, p), Handler)
            break
        except OSError as exc:
            last_err = exc
    if server is None:
        raise SystemExit(
            f"web GUI: no free port in {port}-{port + 19} ({last_err}). "
            f"Pass --port <n> to pick one explicitly."
        )
    bound = server.server_address[1]
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{bound}"
    if bound != port:
        print(
            f"\n  port {port} is busy (gradio/forge-neo or another app uses it) "
            f"— using {bound} instead"
        )
    print(f"\n  Anima LoRA web GUI: {url}\n  (Ctrl-C to stop)\n")
    # Pre-warm the option cache (imports the optimizer zoo) off the request path
    # so the first page load doesn't wait on it.
    threading.Thread(target=options, daemon=True).start()
    if open_browser:
        threading.Thread(
            target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  web GUI stopped.")
        server.shutdown()
