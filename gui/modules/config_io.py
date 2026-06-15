# -*- coding: utf-8 -*-
"""Config save / load for the GUI — the TomlFunctions-equivalent.

Two pure functions (no torch, no Gradio) so they're unit-testable:

* :func:`load_toml_to_form` — a LETS / kohya_ss_anima / anima_lora ``--config_file``
  TOML → the GUI ``form`` dict. Dedicated form fields are populated directly;
  LETS key renames are applied (``timestep_sample_method`` → ``timestep_sampling``,
  ``cache_latents`` → ``use_vae_cache``, ``save_toml`` → ``output_config``,
  ``min/max_timestep`` ÷1000 → ``t_min/t_max``); everything without a dedicated
  field is folded into the ``extra_flags`` string as ``--key value`` tokens so it
  still round-trips and runs.
* :func:`save_form_to_toml` — the GUI ``form`` → a runnable ``--config_file`` TOML.
  Built from the server's real arg builder (``_method_preset_extra``) so the saved
  TOML is exactly what the Start button would launch.
"""

from __future__ import annotations

import tomllib

import toml

# ── TOML key → GUI form field (verbatim string value) ───────────────────────
_DIRECT_FIELDS = {
    "method",
    "preset",
    "optimizer_type",
    "learning_rate",
    "max_train_epochs",
    "seed",
    "network_dim",
    "network_alpha",
    "network_module",
    "sample_prompts",
    "output_name",
    "output_dir",
    "dataset_config",
    "log_every_n_steps",
    "monitor_host",
    "monitor_port",
    "lr_warmup_steps",
    # sd-scripts / LETS training knobs with dedicated GUI fields (Phase 1b)
    "mixed_precision",
    "max_grad_norm",
    "gradient_accumulation_steps",
    "loss_type",
    "huber_c",
    "huber_schedule",
    "timestep_sampling",
    "sigmoid_scale",
    "weighting_scheme",
    "logit_mean",
    "logit_std",
    "attn_mode",
    "blocks_to_swap",
    "t_min",
    "t_max",
    "qwen3_max_token_length",
    "save_every_n_epochs",
    "save_precision",
    "resume",
    # GUI auto-preprocess orchestration knobs (not train.py args — consumed by the
    # server's _prepare_auto_preprocess; mapped here so a load doesn't shove them
    # into extra_flags as bogus train flags).
    "caption_shuffle_variants",
    "caption_tag_dropout_rate",
}
# Boolean form fields (rendered as checkboxes; value kept as bool, not str).
_BOOL_FIELDS = {
    "monitor",
    "gradient_checkpointing",
    "network_train_unet_only",
    "use_vae_cache",
    "use_text_cache",
    "use_shuffled_caption_variants",
    "use_shuffled_caption_variants_only",
    "qwen_image_vae_2d",
    "save_state",
    "output_config",
    # GUI auto-preprocess orchestration toggles (not train.py args).
    "auto_preprocess",
    "multiscale",
    "drop_lowres",
    "mask_enable",
}
# Tri-state dropdown fields ("on"/"off"/blank): a config bool maps to "on"/"off".
_TRISTATE_FIELDS = {"torch_compile"}
# Model-path renames (kohya/LETS name → our form field).
_MODEL_PATHS = {
    "pretrained_model_name_or_path": "dit_path",
    "qwen3": "te_path",
    "vae": "vae_path",
}
# List-valued args → space-joined string in the form field.
_LIST_FIELDS = {"network_args", "optimizer_args", "lr_scheduler_args"}

# LETS / kohya key → our train.py arg name (plain rename; value unchanged).
_RENAME = {
    "timestep_sample_method": "timestep_sampling",
    "cache_latents": "use_vae_cache",
    "cache_latents_to_disk": "use_vae_cache",
    "cache_text_encoder_outputs": "use_text_cache",
    "cache_text_encoder_outputs_to_disk": "use_text_cache",
    "save_toml": "output_config",
}
# Keys with no anima_lora equivalent — dropped on load (documented in the GUI).
_DROP = {
    "train_mode",
    "xformers",
    "prior_loss_weight",
    "max_token_length",
    "no_half_vae",
    "full_fp16",
    "full_bf16",
    # kohya aspect-ratio bucketing — anima_lora uses constant-token tiers
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "bucket_no_upscale",
    "multires_training",
    "resolution",
    "batch_size",
    "lr_scheduler_num_cycles",
    "split_attn",
    "lowram",
}
# Dataset-blueprint sections are not flat scalars — skip the flat-key routing, but
# `datasets` is harvested into the ds_* fields first (see _extract_dataset).
_SKIP_SECTIONS = {"general", "datasets", "subsets"}

# Canonical constant-token resolution tiers (mirror of
# library.datasets.buckets.ALLOWED_TARGET_RES) — kohya `resolution` snaps to the
# nearest one for the Dataset tier checkboxes. Kept inline so config_io stays
# torch-free / import-light.
_DATASET_TIERS = (512, 768, 896, 1024, 1280, 1536)


def _nearest_tier(res) -> int | None:
    """Snap a kohya ``resolution`` (int or [w,h]) to the nearest constant-token tier."""
    if isinstance(res, (list, tuple)):
        res = max(res) if res else None
    try:
        r = float(res)
    except (TypeError, ValueError):
        return None
    return min(_DATASET_TIERS, key=lambda t: abs(t - r))


# Column order of the GUI's "Additional subsets" grid (must match
# gui.kohya.app._DS_EXTRA_COLS) so a multi-subset config round-trips.
_DS_EXTRA_COLS = (
    "image_dir", "cache_dir", "num_repeats", "keep_tokens", "caption_extension",
    "batch_size", "flip_aug", "random_crop", "tiers",
)


def _subset_to_row(s: dict, block_bs=None) -> list:
    """Render one subset dict as an ``_DS_EXTRA_COLS``-ordered grid row."""
    tiers = s.get("tiers")
    return [
        str(s.get("image_dir") or ""),
        str(s.get("cache_dir") or ""),
        s.get("num_repeats"),
        s.get("keep_tokens"),
        str(s.get("caption_extension") or ""),
        s.get("batch_size", block_bs),
        bool(s.get("flip_aug", False)),
        bool(s.get("random_crop", False)),
        ",".join(str(x) for x in tiers) if isinstance(tiers, (list, tuple)) else "",
    ]


def _extract_dataset(data: dict, form: dict) -> None:
    """Harvest the ``[[datasets]]`` blocks into the Dataset panel: the first subset
    fills the flat ``ds_*`` / ``target_res`` fields; any further subsets become rows
    of the ``ds_extra`` grid (so a multi-subset config round-trips). Mutates
    ``form`` in place."""
    blocks = data.get("datasets")
    if not isinstance(blocks, list) or not blocks:
        return
    tiers: set[int] = set()
    flat_bs = None  # batch_size to surface for the first subset (block fallback)
    pairs: list[tuple[dict, object]] = []  # (subset, owning-block batch_size)
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        tier = _nearest_tier(blk.get("resolution"))
        if tier is not None:
            tiers.add(tier)
        subs = blk.get("subsets")
        if isinstance(subs, list):
            for s in subs:
                if isinstance(s, dict):
                    pairs.append((s, blk.get("batch_size")))
    if pairs:
        first, first_block_bs = pairs[0]
        if first.get("image_dir"):
            form["ds_image_dir"] = str(first["image_dir"])
        if first.get("cache_dir"):
            form["ds_cache_dir"] = str(first["cache_dir"])
        for fk, sk in (
            ("ds_num_repeats", "num_repeats"),
            ("ds_keep_tokens", "keep_tokens"),
            ("ds_caption_extension", "caption_extension"),
            ("ds_caption_dropout_rate", "caption_dropout_rate"),
        ):
            if first.get(sk) is not None:
                form[fk] = str(first[sk])
        flat_bs = first.get("batch_size", first_block_bs)
        if flat_bs is not None:
            form["ds_batch_size"] = str(flat_bs)
        for fk, sk in (("ds_flip_aug", "flip_aug"), ("ds_random_crop", "random_crop")):
            if sk in first:
                form[fk] = bool(first[sk])
        t = first.get("tiers")
        if isinstance(t, (list, tuple)) and t:
            form["ds_tiers"] = ",".join(str(x) for x in t)
        if len(pairs) > 1:
            form["ds_extra"] = [_subset_to_row(s, bbs) for s, bbs in pairs[1:]]
    if tiers:
        form["target_res"] = [str(t) for t in sorted(tiers)]


def load_toml_to_form(toml_text: str) -> dict:
    """Parse a config TOML into the GUI ``form`` dict (see module docstring)."""
    data = tomllib.loads(toml_text)
    data.pop("base_config", None)  # inheritance ref — we flatten, ignore it

    form: dict = {}
    extra: list[str] = []  # CLI tokens for the extra_flags field

    # 0) Harvest the dataset blueprint into the ds_* / target_res panel, then drop
    #    the (nested) sections so the flat-key router below never sees them.
    _extract_dataset(data, form)
    for sec in _SKIP_SECTIONS:
        data.pop(sec, None)

    # 1) Normalize LETS/kohya keys into our arg space *before* routing, so renamed
    #    keys land in their dedicated fields (not extra_flags).
    norm: dict = {}
    for key, value in data.items():
        if key == "min_timestep":  # kohya 0–1000 int → flow-matching σ∈[0,1]
            norm["t_min"] = round(float(value) / 1000.0, 6)
        elif key == "max_timestep":
            norm["t_max"] = round(float(value) / 1000.0, 6)
        else:
            norm[_RENAME.get(key, key)] = value

    def emit(flag_key: str, value) -> None:
        """Append a ``--key value`` (or bool) token pair to extra."""
        if isinstance(value, bool):
            extra.append(f"--{flag_key}" if value else f"--no-{flag_key}")
        elif isinstance(value, (list, tuple)):
            extra.append(f"--{flag_key}")
            extra.extend(str(x) for x in value)
        else:
            extra.append(f"--{flag_key}")
            extra.append(str(value))

    # 2) lr_scheduler / lr_scheduler_type both feed the single form field.
    sched = norm.pop("lr_scheduler_type", None) or norm.pop("lr_scheduler", None)
    if sched is not None:
        form["lr_scheduler_type"] = str(sched)

    # 3) Route each normalized key.
    for key, value in norm.items():
        if key in _MODEL_PATHS:
            form[_MODEL_PATHS[key]] = str(value)
        elif key in _LIST_FIELDS:
            form[key] = (
                " ".join(str(x) for x in value)
                if isinstance(value, (list, tuple))
                else str(value)
            )
        elif key in _TRISTATE_FIELDS:
            form[key] = "on" if bool(value) else "off"
        elif key in _BOOL_FIELDS:
            form[key] = bool(value)
        elif key in _DIRECT_FIELDS:
            form[key] = str(value)
        elif key in _DROP:
            continue
        else:
            emit(key, value)

    if extra:
        form["extra_flags"] = " ".join(extra)
    return form


def _argv_to_toml_dict(method: str, preset: str, argv: list[str]) -> dict:
    """Convert a train.py arg list into a TOML-serializable dict."""
    out: dict = {"method": method, "preset": preset}
    i = 0
    n = len(argv)
    while i < n:
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:]
        if key.startswith("no-"):  # --no-flag → false
            out[key[3:]] = False
            i += 1
            continue
        # Gather following non-flag values.
        vals: list[str] = []
        j = i + 1
        while j < n and not argv[j].startswith("--"):
            vals.append(argv[j])
            j += 1
        if not vals:  # bare flag → true
            out[key] = True
        elif len(vals) == 1:
            out[key] = _coerce(vals[0])
        else:
            out[key] = [_coerce(v) for v in vals]
        i = j
    return out


def _coerce(s: str):
    """Best-effort scalar coercion for a CLI value string."""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def save_form_to_toml(form: dict) -> str:
    """The GUI form → a runnable ``--config_file`` TOML (matches Start)."""
    from gui.webgui import server  # pure-stdlib; safe + torch-free

    method, preset, extra = server._method_preset_extra(form)
    d = _argv_to_toml_dict(method, preset, extra)
    header = (
        "# Saved from the Anima GUI — runnable as:\n"
        "#   python train.py --config_file <this file>\n"
    )
    return header + toml.dumps(d)
