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
    "metadata_title",
    "metadata_author",
    "metadata_description",
    "metadata_license",
    "metadata_tags",
    # Accelerate-launch knobs surfaced in the GUI (multi-GPU only). NOT anima train.py
    # args — kept here so a loaded config populates the field and they're NOT shoved
    # into extra_flags (which would crash argparse). The backend ignores them on the
    # default inline path.
    "num_processes",
    "num_machines",
    "num_cpu_threads_per_process",
    "gpu_ids",
    "main_process_port",
    "extra_accelerate_args",
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
    "multi_gpu",
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
# kohya / LoRA_Easy / SD-era keys with NO anima train.py flag — dropped on load so
# they are NOT folded into extra_flags. train.py uses a strict argparse (parse_args,
# not parse_known_args) BEFORE reading the config, so an unrecognized --<key> would
# abort the run at Start with "unrecognized arguments". Mirror of the import-side drop
# set gui/backend.py::_IMPORT_DROP — keep the two in sync. (Anima-VALID args like
# prior_loss_weight / no_half_vae are NOT here — they round-trip via emit()/_EMIT_IF_TRUE.)
_DROP = {
    "train_mode",
    "xformers",
    "max_token_length",
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
    # SD1.5/SDXL model-family + meta keys with no anima equivalent (anima is a
    # flow-matching DiT + Qwen3 TE). These appear in essentially every kohya /
    # LoRA_Easy config and otherwise crash argparse on Start. skip_image_resolution
    # is handled by choose_edge/target_res; (caption) shuffle is a preprocess-time
    # knob (caption_shuffle_variants), a train-time no-op.
    "skip_image_resolution",
    "shuffle_caption",
    "sdxl",
    "v2",
    "v_parameterization",
    "clip_skip",
    "name",
    # Consumed by _extract_dataset (restores the per-subset gradient_checkpointing
    # checkboxes by tier) — dropped here so it isn't ALSO folded into extra_flags,
    # which would double-own the flag with the subset blocks on the next save.
    "gradient_checkpointing_resolutions",
    # LoRA_Easy_Training_Scripts-only keys with NO anima train.py equivalent — drop
    # them (don't fold into extra_flags, which would crash argparse). save_toml itself
    # → output_config; anima writes the .snapshot.toml beside the checkpoint, so there
    # is no separate "location". The log/run-name modes + edm2 loss are LETS-specific.
    "save_toml_location",
    "log_prefix_mode",
    "run_name_mode",
    "edm2_loss_weighting",
    # Derived at save from output_dir/output_name (logging_dir = <out>/log) — drop on
    # load so it isn't folded into extra_flags as a stray --logging_dir token.
    "logging_dir",
}
# Anima-VALID store_true flags with no dedicated GUI field: emit `--flag` ONLY when
# truthy. A plain store_true has no `--no-<flag>` form, so routing a false value through
# emit() (which would write `--no-no_half_vae`) would itself crash argparse — hence the
# special case rather than membership in _DROP. (prior_loss_weight is a float and rides
# the normal emit() path; lowram is owned by _BOOL_FIELDS, checked before _DROP.)
_EMIT_IF_TRUE = {"no_half_vae"}
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


# Per-subset field mapping for the legacy flat-config round-trip (configs without a
# [[datasets]] blueprint): the primary subset uses the "ds_" prefix, extras
# "ds2_"/"ds3_"/…. The native GUI's dynamic subset cards round-trip via form["subsets"];
# these prefixes remain for importing older flat configs.
_DS_N_SUBSETS = 4
_DS_PREFIXES = ("ds_",) + tuple(f"ds{i}_" for i in range(2, _DS_N_SUBSETS + 1))
_DS_STR_FIELDS = (
    ("num_repeats", "num_repeats"),
    ("keep_tokens", "keep_tokens"),
    ("caption_extension", "caption_extension"),
    ("caption_dropout_rate", "caption_dropout_rate"),
)
_DS_BOOL_FIELDS = (
    ("flip_aug", "flip_aug"),
    ("random_crop", "random_crop"),
    ("gradient_checkpointing", "gradient_checkpointing"),
)


def _fill_subset_block(
    form: dict, prefix: str, s: dict, block_bs, gc_edges: set
) -> None:
    """Write one subset dict into the form's prefixed block fields. cache_dir lands
    only on the primary (extras share it in the GUI). gradient_checkpointing is
    reconstructed from a tier match against ``gradient_checkpointing_resolutions`` (the
    backend's per-subset GC emit — the value isn't stored as a subset TOML key)."""
    if s.get("image_dir"):
        form[f"{prefix}image_dir"] = str(s["image_dir"])
    if prefix == "ds_" and s.get("cache_dir"):
        form["ds_cache_dir"] = str(s["cache_dir"])
    for fk, sk in _DS_STR_FIELDS:
        if s.get(sk) is not None:
            form[f"{prefix}{fk}"] = str(s[sk])
    bs = s.get("batch_size", block_bs)
    if bs is not None:
        form[f"{prefix}batch_size"] = str(bs)
    for fk, sk in _DS_BOOL_FIELDS:
        if sk in s:
            form[f"{prefix}{fk}"] = bool(s[sk])
    t = s.get("tiers")
    tier_ints: set[int] = set()
    if isinstance(t, (list, tuple)) and t:
        form[f"{prefix}tiers"] = ",".join(str(x) for x in t)
        tier_ints = {int(x) for x in t if str(x).isdigit()}
    if gc_edges and tier_ints and (tier_ints & gc_edges):
        form[f"{prefix}gradient_checkpointing"] = True


def _extract_native_subsets(data: dict, form: dict) -> None:
    """Rebuild the native GUI's dynamic subset cards (``form['subsets']``) from the
    ``[[datasets]]`` blueprint a GUI save writes (see save_form_to_toml). Values are
    stringified for the card line-edits; flip_aug/random_crop/is_val stay bool; the
    per-subset gradient_checkpointing toggle is reconstructed from a tier match against
    ``gradient_checkpointing_resolutions`` (it isn't stored as a subset key). Mutates
    ``form`` in place; sets nothing when there are no subset blocks (legacy flat configs
    still fall through to _extract_dataset's ds_* path)."""
    blocks = data.get("datasets")
    if not isinstance(blocks, list) or not blocks:
        return
    gc_edges = {
        int(x)
        for x in (data.get("gradient_checkpointing_resolutions") or [])
        if str(x).isdigit()
    }
    out: list[dict] = []
    seen: set[str] = set()  # dedup same-folder subsets (kohya multiscale lists 1/res)
    for blk in blocks:
        if not isinstance(blk, dict):
            continue
        blk_bs = blk.get("batch_size")
        for s in blk.get("subsets") or []:
            if not isinstance(s, dict):
                continue
            img = str(s.get("image_dir") or "").strip()
            if not img or img in seen:
                continue
            seen.add(img)
            row: dict = {"image_dir": img}
            if str(s.get("cache_dir") or "").strip():
                row["cache_dir"] = str(s["cache_dir"]).strip()
            for k in (
                "num_repeats",
                "keep_tokens",
                "caption_dropout_rate",
                "caption_extension",
            ):
                if s.get(k) is not None:
                    row[k] = str(s[k])
            bs = s.get("batch_size", blk_bs)
            if bs is not None:
                row["batch_size"] = str(bs)
            tier_ints: set[int] = set()
            t = s.get("tiers")
            if isinstance(t, (list, tuple)) and t:
                row["tiers"] = ",".join(str(x) for x in t)
                tier_ints = {int(x) for x in t if str(x).isdigit()}
            for fk in ("flip_aug", "random_crop", "is_val"):
                if s.get(fk):
                    row[fk] = True
            if gc_edges and tier_ints and (tier_ints & gc_edges):
                row["gradient_checkpointing"] = True
            out.append(row)
    if out:
        form["subsets"] = out


def _extract_dataset(data: dict, form: dict) -> None:
    """Harvest the ``[[datasets]]`` blocks into the per-subset block fields: the first
    subset fills ds_*, further subsets ds2_*/ds3_*/… (up to N). Resolutions across
    blocks seed target_res; gradient_checkpointing_resolutions restores the per-subset
    GC checkboxes by tier. Mutates ``form`` in place."""
    blocks = data.get("datasets")
    if not isinstance(blocks, list) or not blocks:
        return
    gc_edges = {
        int(x)
        for x in (data.get("gradient_checkpointing_resolutions") or [])
        if str(x).isdigit()
    }
    tiers: set[int] = set()
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
    # Dedup by image_dir: a LoRA_Easy multiscale dataset_config lists the SAME folder
    # under N [[datasets]] blocks (one per resolution). Anima expresses that as ONE
    # subset + the union of tiers in target_res (choose_edge assigns each image to the
    # tier that resizes it least) — so collapse same-folder subsets to the first.
    seen: set = set()
    deduped: list = []
    for s, bs in pairs:
        img = str(s.get("image_dir") or "").strip()
        if img and img in seen:
            continue
        if img:
            seen.add(img)
        deduped.append((s, bs))
    for prefix, (s, block_bs) in zip(_DS_PREFIXES, deduped):
        _fill_subset_block(form, prefix, s, block_bs, gc_edges)
    if len(deduped) > _DS_N_SUBSETS:
        form["_ds_overflow"] = len(deduped)  # surfaced as a load note by on_load_config
    if tiers:
        form["target_res"] = [str(t) for t in sorted(tiers)]


def load_toml_to_form(toml_text: str, known_dests=None) -> dict:
    """Parse a config TOML into the GUI ``form`` dict (see module docstring).

    ``known_dests`` (optional): a set of arg dests the caller renders as real form
    fields beyond the curated ``_DIRECT_FIELDS`` set — e.g. the native GUI's full
    schema-arg fields (``backend.list_arg_groups``). Any such key passes through to
    ``form[key]`` (raw value) instead of being folded into ``extra_flags``, so a
    loaded config populates its dropdown / field rather than the catch-all box.
    """
    known = set(known_dests or ())
    data = tomllib.loads(toml_text)
    data.pop("base_config", None)  # inheritance ref — we flatten, ignore it

    form: dict = {}
    extra: list[str] = []  # CLI tokens for the extra_flags field

    # 0) Harvest the dataset blueprint: first into the native GUI's dynamic subset
    #    cards (form['subsets']), then the legacy ds_* / target_res panel, then drop
    #    the (nested) sections so the flat-key router below never sees them.
    _extract_native_subsets(data, form)
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
            # network/optimizer/lr_scheduler args → one `key=value` per LINE for the
            # multi-line textbox editor (the backend's _arg_split treats newlines as
            # whitespace, so the inline form still parses). Handles arbitrary length.
            form[key] = (
                "\n".join(str(x) for x in value)
                if isinstance(value, (list, tuple))
                else str(value)
            )
        elif key in _TRISTATE_FIELDS:
            form[key] = "on" if bool(value) else "off"
        elif key in _BOOL_FIELDS:
            form[key] = bool(value)
        elif key in _DIRECT_FIELDS:
            form[key] = str(value)
        elif key in _EMIT_IF_TRUE:
            if value:  # store_true: only --flag (no --no-flag form) — false ⇒ drop
                emit(key, True)
        elif key in _DROP:
            continue
        elif key in known:
            # A schema/advanced field the caller renders directly (native GUI):
            # pass the raw value through so it populates the field, not extra_flags.
            form[key] = value
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
    # Serialize the dynamic subset cards as a [[datasets]] blueprint — _method_preset_extra
    # consumes form['subsets'] only for --gradient_checkpointing_resolutions, so without
    # this every per-subset field (image_dir, caption_extension, flip_aug/random_crop/
    # is_val, num_repeats…) would be dropped and the cards couldn't be restored on load.
    subs = server._dataset_subsets(form)
    if subs:
        d["datasets"] = [{"subsets": subs}]
    header = (
        "# Saved from the Anima GUI — runnable as:\n"
        "#   python train.py --config_file <this file>\n"
    )
    return header + toml.dumps(d)
