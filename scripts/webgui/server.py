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
    "dataset_config",
    "max_train_epochs",
    "output_name",
    "seed",
    "monitor",
    "monitor_host",
    "monitor_port",
    "monitor_open_browser",
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
            "gradient_checkpoint", "gradient_accumulation",
            "max_data_loader", "train_batch_size", "max_train_epochs",
            "max_train_steps", "prior_loss_weight", "lowram", "highvram",
            "compile", "dynamo", "cudagraph", "activation_memory",
            "attn_mode", "attn_softmax", "flash", "sdpa", "sageattn", "flex",
            "blocks_to_swap", "block_swap", "channel_scal",
            "persistent_data", "pin_memory", "prefetch", "dataloader",
            "split_attn", "vae_chunk", "vae_disable_cache", "vae_batch_size",
            "unsloth", "cpu_offload", "fused_backward", "skip_until", "initial_",
        ],
    ),
    (
        "NETWORK",  # adapter dropout/train-on/weights · timestep window · net regularization
        [
            "network",  # network_dropout / network_train_* / network_weights
            "t_min", "t_max", "scale_weight", "base_weights",
            "lora_path", "lora_multiplier", "dim_from_weights",
        ],
    ),
    (
        "OPTIMIZER",  # optimizer · scheduler · LR · loss
        [
            "optimizer", "unet_lr", "text_encoder_lr", "scheduler",
            "lr_warmup", "lr_decay", "loss_type", "masked_loss", "huber",
            "min_snr", "multiscale_loss", "debiased", "max_grad_norm",
        ],
    ),
    ("SAVE", ["save", "resume", "state", "config_snapshot", "output_dir", "output_config", "metadata", "checkpointing"]),
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
        "ANIMA",  # flow-matching · tokenizer · Anima-specific experimental features
        [
            "timestep", "sigmoid", "weighting", "discrete_flow", "shift",
            "logit", "mode_scale", "t5", "qwen", "tokenizer",
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
        flag = max(a.option_strings, key=len)  # the long --form
        is_bool = _arg_type(a) == "bool"
        item = {
            "dest": dest,
            "flag": flag,
            "type": _arg_type(a),
            "is_bool": is_bool,
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
        {"role": label.get(r, r), "args": sorted(buckets[r], key=lambda x: x["dest"])}
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
    """LyCORIS target presets (network_args preset=...).

    The ``anima-*`` entries are the ones that actually wrap the Anima DiT
    (anima-attn-mlp → 197 modules, attention+MLP; anima-full → 314, +adaln/embeds).
    The stock built-ins target standard diffusers class names and are kept only
    for non-Anima base models."""
    return [
        *_ANIMA_LYCORIS_PRESETS,
        "full",
        "full-lin",
        "attn-mlp",
        "attn-only",
        "unet-transformer-only",
        "unet-convblock-only",
    ]


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
    add("--output_name", "output_name")
    add("--seed", "seed")

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
    if nm:
        extra += ["--network_module", nm]
    na = str(form.get("network_alpha", "")).strip()
    if na:
        extra += ["--network_alpha", na]
    nargs = (form.get("network_args") or "").split()
    if "lycoris" in nm:  # lycoris.kohya OR networks.lycoris_anima (the Anima bridge)
        lp = (form.get("lycoris_preset") or "").strip()
        if lp:
            # Friendly Anima preset names → shipped TOML paths; pass others verbatim.
            lp = _ANIMA_LYCORIS_PRESETS.get(lp, lp)
            nargs = [f"preset={lp}"] + nargs
        algo = (form.get("algo") or "").strip()
        if algo:
            nargs = [f"algo={algo}"] + nargs
    if nargs:
        extra += ["--network_args", *nargs]

    # Auto-generated "all arguments" section: each enabled item is
    # {flag, value, is_bool, nargs}. Only enabled args are emitted (the rest fall
    # back to the config-chain defaults).
    for item in form.get("adv") or []:
        flag = item.get("flag")
        if not flag:
            continue
        if item.get("is_bool"):
            if item.get("value"):
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
    port = str(form.get("monitor_port") or "8765")
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

    name = _safe_name(form.get("ds_name") or form.get("output_name") or "gui")
    base_resized = f"post_image_dataset/resized/{name}"
    base_cache = f"post_image_dataset/lora/{name}"
    base_mask = f"post_image_dataset/masks/{name}"
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

    env = {"MANIFEST_FILE": str(mf_path)}
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
    }


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

            cl = _dc.ensure_daemon()
            if prep:
                # preprocess command-job → auto-chains the train job on success
                # (manager._finalize). One Start click; both phases survive close.
                resp = cl.submit_command(
                    label=prep["target"],
                    argv=prep["argv"],
                    extra_env=prep["extra_env"],
                    chain_train={"method": method, "preset": preset, "extra": extra},
                )
            else:
                resp = cl.submit(method=method, preset=preset, extra=extra)
            _STATE.update(
                proc=None,
                cmd=None,
                started_at=None,
                monitor_url=mon,
                daemon_job=resp.get("job_id"),
                daemon_base=getattr(cl, "base", None),
            )
            return {
                "ok": True,
                "daemon": True,
                "job_id": resp.get("job_id"),
                "daemon_base": getattr(cl, "base", None),
                "monitor_url": mon,
                "preprocess": bool(prep),
                "command": cmd_str,
                "note": ("auto-preprocess → train chain submitted" if prep else None),
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


def status() -> dict:
    proc = _STATE.get("proc")
    running = proc is not None and proc.poll() is None
    return {
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


def stop() -> dict:
    proc = _STATE.get("proc")
    if proc is None or proc.poll() is not None:
        return {"ok": False, "error": "no run in progress"}
    try:
        proc.terminate()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


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
    if isinstance(v, dict):
        return " ".join(f"{k}={vv}" for k, vv in v.items())
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
# constant-token buckets + preprocess-time caption shuffle, not these).
_IMPORT_DROP = {
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "bucket_no_upscale",
    "multires_training",
    "skip_image_resolution",
    "shuffle_caption",
    "caption_tag_dropout_rate",
    "sdxl",
    "v2",
    "v_parameterization",
    "clip_skip",
    "xformers",
    "no_half_vae",
    "min_snr_gamma",
    "prior_loss_weight",
    "reg_data_dir",
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

    # Dataset blueprint (anima/kohya): pull subsets from the [[datasets]] blocks.
    res_for_tier = flat.get("resolution")
    if datasets and not subsets:
        for ds in datasets:
            if isinstance(ds, dict):
                res_for_tier = res_for_tier or ds.get("resolution")
                subsets.extend(ds.get("subsets") or [])

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
    _put("seed", "seed")

    # network module / algo / preset / extra net args
    nm = str(flat.get("network_module") or "").strip()
    na = flat.get("network_args")
    na_dict = dict(na) if isinstance(na, dict) else {}
    if isinstance(na, (list, tuple)):
        for item in na:
            if isinstance(item, str) and "=" in item:
                k, v = item.split("=", 1)
                na_dict[k] = v
    if nm:
        form["network_module"] = nm if "lycoris" in nm else "networks.lora_anima"
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

    # resolution → target_res tier(s)
    tier = _snap_tier(res_for_tier) if res_for_tier is not None else None
    if tier:
        form["target_res"] = [str(tier)]
        notes.append(
            f"resolution {res_for_tier} → target_res tier {tier} (anima constant-token)."
        )

    # subsets → the manual builder's shape (only valid anima keys)
    out_subs = []
    for s in subsets:
        if not isinstance(s, dict) or not s.get("image_dir"):
            continue
        out_subs.append(
            {
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
        )

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
