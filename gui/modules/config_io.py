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
    "constantcosine_tail_epochs",
    "lr_scheduler_min_lr_ratio",
    # Sample-image cadence (curated in the "Sample images" panel).
    "sample_every_n_steps",
    "sample_every_n_epochs",
    "sample_sampler",
    # Promoted kohya-parity knobs (now curated first-class fields).
    "train_batch_size",
    "max_train_steps",
    "network_weights",
    "unet_lr",
    "llm_adapter_lr",
    "text_encoder_lr",
    "lr_scheduler_num_cycles",
    "lr_scheduler_power",
    "scale_weight_norms",
    "network_dropout",
    "save_every_n_steps",
    "save_last_n_steps",
    "save_last_n_steps_state",
    "save_last_n_epochs",
    "save_last_n_epochs_state",
    "max_data_loader_n_workers",
    "vae_batch_size",
    "training_comment",
    "log_with",
    "wandb_run_name",
    "wandb_api_key",
    "log_tracker_name",
    "save_model_as",
    "t5_max_token_length",
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
    "use_constantcosine",
    "save_state",
    "save_state_on_train_end",
    "sample_at_first",
    "dim_from_weights",
    "highvram",
    "lowram",
    "persistent_data_loader_workers",
    "vae_disable_cache",
    "output_config",
    # GUI auto-preprocess orchestration toggles (not train.py args).
    "auto_preprocess",
    "multiscale",
    "drop_lowres",
    "mask_enable",
}
# Tri-state dropdown fields ("on"/"off"/blank): a config bool maps to "on"/"off".
# torch_compile/masked_loss/skip_cache_check all default ON, so a plain checkbox
# couldn't force them off — the GUI renders them as on/off/blank dropdowns.
_TRISTATE_FIELDS = {"torch_compile", "masked_loss", "skip_cache_check"}
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
    # Consumed by _extract_dataset (restores the per-subset gradient_checkpointing
    # checkboxes by tier) — dropped here so it isn't ALSO folded into extra_flags,
    # which would double-own the flag with the subset blocks on the next save.
    "gradient_checkpointing_resolutions",
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


# Per-subset scalar/bool keys carried verbatim into each subset dict.
_DS_STR_FIELDS = (
    "num_repeats",
    "keep_tokens",
    "caption_extension",
    "caption_dropout_rate",
)
_DS_BOOL_FIELDS = ("flip_aug", "random_crop", "gradient_checkpointing")


def _extract_dataset(data: dict, form: dict) -> None:
    """Harvest the ``[[datasets]]`` blocks into ``form['subsets']`` = list[dict] — the
    shape the GUI's dynamic subset rows (and the backend) consume. Each block's
    batch_size falls through to its subsets; resolutions across blocks seed target_res;
    gradient_checkpointing_resolutions restores the per-subset GC flag by tier match.
    The first subset's cache_dir is surfaced as the shared ds_cache_dir. Mutates
    ``form`` in place."""
    blocks = data.get("datasets")
    if not isinstance(blocks, list) or not blocks:
        return
    gc_edges = {
        int(x)
        for x in (data.get("gradient_checkpointing_resolutions") or [])
        if str(x).isdigit()
    }
    tiers: set[int] = set()
    subsets: list[dict] = []
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        tier = _nearest_tier(blk.get("resolution"))
        if tier is not None:
            tiers.add(tier)
        block_bs = blk.get("batch_size")
        for s in blk.get("subsets") or []:
            if not isinstance(s, dict) or not s.get("image_dir"):
                continue
            sub: dict = {"image_dir": str(s["image_dir"])}
            if s.get("cache_dir"):
                sub["cache_dir"] = str(s["cache_dir"])
            for k in _DS_STR_FIELDS:
                if s.get(k) is not None:
                    sub[k] = s[k]
            bs = s.get("batch_size", block_bs)
            if bs is not None:
                sub["batch_size"] = bs
            for k in _DS_BOOL_FIELDS:
                if k in s:
                    sub[k] = bool(s[k])
            t = s.get("tiers")
            tier_ints: set[int] = set()
            if isinstance(t, (list, tuple)) and t:
                sub["tiers"] = list(t)
                tier_ints = {int(x) for x in t if str(x).isdigit()}
            if gc_edges and tier_ints and (tier_ints & gc_edges):
                sub["gradient_checkpointing"] = True
            subsets.append(sub)
    if subsets:
        form["subsets"] = subsets
        # surface the (shared) cache_dir from the first subset that has one
        cd = next((s["cache_dir"] for s in subsets if s.get("cache_dir")), None)
        if cd:
            form["ds_cache_dir"] = str(cd)
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
    from gui import backend as server  # pure-stdlib; safe + torch-free

    method, preset, extra = server._method_preset_extra(form)
    d = _argv_to_toml_dict(method, preset, extra)
    header = (
        "# Saved from the Anima GUI — runnable as:\n"
        "#   python train.py --config_file <this file>\n"
    )
    return header + toml.dumps(d)
