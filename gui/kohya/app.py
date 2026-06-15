# -*- coding: utf-8 -*-
"""Gradio Blocks UI for the Anima LoRA trainer.

Layout modelled on the kohya_ss GUI (source-model paths + an Anima accordion +
basic / network / optimizer / dataset / sample / advanced sections), but every
control feeds the ``form`` dict consumed by :mod:`gui.backend`, whose
``build_command`` emits the exact ``train.py --method … --preset …`` invocation
and whose ``launch`` runs it as a direct subprocess. No kohya ``sd-scripts``
code path is involved — the GUI is a front-end re-skin over our own backend.
"""

from __future__ import annotations

from gui import backend as server
from gui.modules.arg_help import ARG_HELP
from gui.modules.config_io import load_toml_to_form, save_form_to_toml

# --------------------------------------------------------------------------- #
# Form-field registry: the click handlers receive positional values, so the
# order of FIELD_KEYS MUST match the order components are appended to `inputs`.
# Each value lands verbatim in the `form` dict server._method_preset_extra /
# server.launch already understand — blanks fall back to the config chain.
# --------------------------------------------------------------------------- #
FIELD_KEYS: list[str] = []


def _register(keys: list[str], comps: list, key: str, comp):
    """Append one component, recording its form key in lockstep. Attaches the
    ported Korean per-field help (ARG_HELP) as the component's `info` tooltip when
    the field has one and none was set explicitly — Gradio serializes `info` from
    get_config() at render, so setting it post-construction is honoured."""
    keys.append(key)
    comps.append(comp)
    help_text = ARG_HELP.get(key)
    if help_text and not getattr(comp, "info", None):
        try:
            comp.info = help_text
        except Exception:  # noqa: BLE001  (info is best-effort cosmetic)
            pass
    return comp


def _collect(keys: list[str], values) -> dict:
    """Zip positional handler args back into the backend `form` dict."""
    form = dict(zip(keys, values))
    # Gradio Textbox yields "" for empty; backend treats "" as "use default".
    # Coerce checkbox-style truthiness through untouched (already bool).
    _assemble_dataset(form)
    return form


# Flat dataset-field key → (subset-dict key, caster) for the PRIMARY subset (the
# common single-folder case). The backend's _dataset_subsets list-branch consumes
# the assembled form["subsets"]; additional subsets come from the ds_extra grid.
_DS_SUBSET_FIELDS = (
    ("ds_num_repeats", "num_repeats", int),
    ("ds_keep_tokens", "keep_tokens", int),
    ("ds_caption_extension", "caption_extension", str),
    ("ds_caption_dropout_rate", "caption_dropout_rate", float),
    ("ds_batch_size", "batch_size", int),
)
_DS_SUBSET_BOOLS = (("ds_flip_aug", "flip_aug"), ("ds_random_crop", "random_crop"))
# Column order of the "Additional subsets" gr.Dataframe (type="array" → list rows).
# Shared with config_io._DS_EXTRA_COLS so the load round-trip stays symmetric.
_DS_EXTRA_COLS = (
    "image_dir", "cache_dir", "num_repeats", "keep_tokens", "caption_extension",
    "batch_size", "flip_aug", "random_crop", "tiers",
)
_DS_POP_KEYS = (
    "ds_image_dir", "ds_cache_dir", "ds_tiers", "ds_extra",
    *(k for k, _s, _c in _DS_SUBSET_FIELDS),
    *(k for k, _s in _DS_SUBSET_BOOLS),
)


def _tier_list(text) -> list[int]:
    """Parse a free-form tier string (\"512,1024\" / \"1024\") into ints."""
    import re

    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


def _parse_extra_subsets(rows) -> list[dict]:
    """Turn the ds_extra grid (list-of-rows in _DS_EXTRA_COLS order) into subset
    dicts. Rows without an image_dir are skipped (e.g. the trailing blank row)."""
    out: list[dict] = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)):
            continue
        cells = dict(zip(_DS_EXTRA_COLS, row))
        img = str(cells.get("image_dir") or "").strip()
        if not img:
            continue
        sub: dict = {"image_dir": img}
        cache = str(cells.get("cache_dir") or "").strip()
        if cache:
            sub["cache_dir"] = cache
        for k in ("num_repeats", "keep_tokens", "batch_size"):
            v = cells.get(k)
            if v in (None, ""):
                continue
            try:
                sub[k] = int(float(v))
            except (TypeError, ValueError):
                pass
        ce = str(cells.get("caption_extension") or "").strip()
        if ce:
            sub["caption_extension"] = ce
        for k in ("flip_aug", "random_crop"):
            if cells.get(k) in (True, "true", "True", 1, "1"):
                sub[k] = True
        tiers = _tier_list(cells.get("tiers"))
        if tiers:
            sub["tiers"] = tiers
        out.append(sub)
    return out


def _assemble_dataset(form: dict) -> None:
    """Fold the flat ds_* primary-subset fields + the ds_extra grid into
    ``form['subsets']`` (primary first, then extra rows) so the backend builds a
    precached dataset config. No-op when nothing has an image_dir (defer to the
    base.toml blueprint / an explicit --dataset_config). Mutates ``form``."""
    subs: list[dict] = []
    img = str(form.get("ds_image_dir") or "").strip()
    if img:
        sub: dict = {"image_dir": img}
        cache = str(form.get("ds_cache_dir") or "").strip()
        if cache:
            sub["cache_dir"] = cache
        for fk, sk, cast in _DS_SUBSET_FIELDS:
            v = form.get(fk)
            if v in (None, ""):
                continue
            try:
                sub[sk] = cast(v)
            except (TypeError, ValueError):
                pass
        for fk, sk in _DS_SUBSET_BOOLS:
            if form.get(fk):
                sub[sk] = True
        tiers = _tier_list(form.get("ds_tiers"))
        if tiers:
            sub["tiers"] = tiers
        subs.append(sub)
    subs.extend(_parse_extra_subsets(form.get("ds_extra")))
    if subs and not form.get("subsets"):
        form["subsets"] = subs
    for k in _DS_POP_KEYS:  # don't leak the flat helpers to the backend form
        form.pop(k, None)


def _pick_path(current: str, *, file: bool) -> str:
    """Open a NATIVE folder/file dialog on the local machine and return the chosen path,
    or the current value on cancel / headless / error. The Gradio GUI runs locally, so the
    dialog pops on the user's own desktop (the kohya_ss browse-button pattern). Fully
    guarded — a failure just keeps whatever was typed, so manual entry always still works.
    Set ANIMA_GUI_NO_PICKER=1 (or run headless) to disable."""
    import os
    import sys

    current = (current or "").strip()
    if os.environ.get("ANIMA_GUI_NO_PICKER"):
        return current
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return current  # no X display — can't pop a dialog
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        initdir = current if os.path.isdir(current) else (os.path.dirname(current) or os.getcwd())
        if file:
            chosen = filedialog.askopenfilename(initialdir=initdir)
        else:
            chosen = filedialog.askdirectory(initialdir=initdir)
        root.destroy()
        return chosen or current
    except Exception:
        return current


def _pick_dir(current):
    return _pick_path(current, file=False)


def _pick_file(current):
    return _pick_path(current, file=True)


def build_app(default_port: int = 7860):
    """Construct the Gradio Blocks app. Imports gradio lazily so importing this
    module (and thus tasks.py) never requires the optional dep."""
    import gradio as gr

    opts = server.options()
    methods = opts["methods"]
    presets = opts["presets"]
    optimizers = opts["optimizers"]
    schedulers = [""] + opts["schedulers"]
    network_modules = [""] + opts["network_modules"]

    keys: list[str] = []
    inputs: list = []
    by_key: dict = {}  # field name → component, for cross-field dependency greying

    def reg(key, comp):
        by_key[key] = comp
        return _register(keys, inputs, key, comp)

    def reg_path(key, *, file=False, **tb_kwargs):
        """Register a path Textbox + a 📁 browse button (native local folder/file dialog).
        ``file=True`` picks a file, else a directory. Lays them side by side in a Row."""
        with gr.Row():
            tb = gr.Textbox(scale=8, **tb_kwargs)
            btn = gr.Button("📁", scale=0, min_width=46)
        reg(key, tb)
        btn.click(_pick_file if file else _pick_dir, inputs=tb, outputs=tb)
        return tb

    with gr.Blocks(title="Anima LoRA Trainer") as demo:
        gr.Markdown(
            "# Anima LoRA Trainer\n"
            "Gradio front-end (kohya-style layout) driving **this repo's** "
            "`train.py --method <name> --preset <name>`. Blank fields defer to the "
            "`base.toml → preset → method` config chain. Start runs train.py "
            "directly; its log streams below and to the terminal."
        )

        with gr.Tab("LoRA"):
            # ── Method / preset (this repo's core training-selection concept) ──
            with gr.Row():
                reg(
                    "method",
                    gr.Dropdown(
                        methods,
                        value=(methods[0] if methods else "lora"),
                        label="Method (configs/methods/<name>.toml)",
                    ),
                )
                reg(
                    "preset",
                    gr.Dropdown(
                        presets,
                        value=(
                            "default"
                            if "default" in presets
                            else (presets[0] if presets else "default")
                        ),
                        label="Hardware preset (configs/presets.toml)",
                    ),
                )

            # ── Output folders ──────────────────────────────────────────────
            with gr.Accordion("Output", open=True):
                with gr.Row():
                    reg(
                        "output_name",
                        gr.Textbox(
                            label="Output name",
                            placeholder="(defaults to method name)",
                        ),
                    )
                    reg_path(
                        "output_dir",
                        value="output",
                        label="Output base dir",
                    )

            # ── Anima model paths (mirrors kohya's class_anima accordion) ────
            with gr.Accordion(
                "Anima Model Paths (blank = config-chain default)", open=False
            ):
                reg_path(
                    "dit_path",
                    file=True,
                    label="DiT checkpoint (--pretrained_model_name_or_path)",
                    placeholder="Path to the Anima DiT .safetensors",
                )
                reg_path(
                    "te_path",
                    file=True,
                    label="Qwen3 text encoder (--qwen3)",
                    placeholder="Path to Qwen3-0.6B model dir / .safetensors",
                )
                reg_path(
                    "vae_path",
                    file=True,
                    label="VAE (--vae)",
                    placeholder="Path to the Qwen-Image VAE",
                )

            # ── Basic training params ───────────────────────────────────────
            with gr.Accordion("Basic", open=True):
                with gr.Row():
                    reg(
                        "learning_rate",
                        gr.Textbox(
                            label="Learning rate", placeholder="(method default)"
                        ),
                    )
                    reg(
                        "max_train_epochs",
                        gr.Textbox(
                            label="Max train epochs", placeholder="(method default)"
                        ),
                    )
                    reg("seed", gr.Textbox(label="Seed", placeholder="(random)"))
                with gr.Row():
                    reg(
                        "network_dim",
                        gr.Textbox(
                            label="Network dim / rank", placeholder="(method default)"
                        ),
                    )
                    reg(
                        "network_alpha",
                        gr.Textbox(
                            label="Network alpha", placeholder="(method default)"
                        ),
                    )

            # ── Optimizer & scheduler ───────────────────────────────────────
            with gr.Accordion("Optimizer & Scheduler", open=False):
                with gr.Row():
                    reg(
                        "optimizer_type",
                        gr.Dropdown(
                            optimizers,
                            value=(optimizers[0] if optimizers else "AdamW"),
                            label="Optimizer (kohya built-ins + vendored zoo)",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "lr_scheduler_type",
                        gr.Dropdown(
                            schedulers,
                            value="",
                            label="LR scheduler (blank = config default)",
                            allow_custom_value=True,
                        ),
                    )
                with gr.Row():
                    reg(
                        "lr_warmup_steps",
                        gr.Textbox(label="LR warmup steps", placeholder="(default)"),
                    )
                    reg(
                        "optimizer_args",
                        gr.Textbox(
                            label="optimizer_args (k=v …)",
                            placeholder="weight_decay=0.01 betas=0.9,0.99",
                        ),
                    )
                    reg(
                        "lr_scheduler_args",
                        gr.Textbox(
                            label="lr_scheduler_args (k=v …)",
                            placeholder="",
                        ),
                    )
                # constant→cosine one-shot: hold constant LR for the planned run,
                # then extend with N cosine-decay epochs (LR→floor) in the SAME run.
                # Overrides lr_scheduler (which greys out when this is on).
                with gr.Row():
                    reg("use_constantcosine", gr.Checkbox(
                        value=False, label="use_constantcosine (constant→cosine)"))
                    reg("constantcosine_tail_epochs", gr.Textbox(
                        label="constantcosine_tail_epochs", placeholder="0 = off"))
                    reg("lr_scheduler_min_lr_ratio", gr.Textbox(
                        label="min_lr_ratio (cosine floor)", placeholder="0.0"))

            # ── Network / adapter ───────────────────────────────────────────
            with gr.Accordion("Network / Adapter", open=False):
                reg(
                    "network_module",
                    gr.Dropdown(
                        network_modules,
                        value="",
                        label="network_module (blank = method default)",
                        allow_custom_value=True,
                    ),
                )
                reg(
                    "network_args",
                    gr.Textbox(
                        label="network_args (k=v …)",
                        placeholder="conv_dim=8 conv_alpha=4",
                    ),
                )

            # ── Dataset ─────────────────────────────────────────────────────
            with gr.Accordion("Dataset", open=True):
                gr.Markdown(
                    "Define **one subset** by folder (the common single-character "
                    "case) — the trainer builds a precached `--dataset_config` from "
                    "it on Start. Run `make preprocess` **first**: point `image_dir` "
                    "at the resized+cached dir (or set `cache_dir`). For multiple "
                    "subsets, use a `--dataset_config` TOML below or the stdlib web GUI."
                )
                with gr.Row():
                    reg_path("ds_image_dir",
                        label="image_dir (resized + cached images)",
                        placeholder="post_image_dataset/resized")
                    reg_path("ds_cache_dir",
                        label="cache_dir (pre-cached latents; blank = default)",
                        placeholder="post_image_dataset/lora")
                with gr.Row():
                    reg("ds_num_repeats", gr.Textbox(
                        label="num_repeats", placeholder="1"))
                    reg("ds_keep_tokens", gr.Textbox(
                        label="keep_tokens", placeholder="0"))
                    reg("ds_caption_extension", gr.Textbox(
                        label="caption_extension", placeholder=".txt"))
                    reg("ds_batch_size", gr.Textbox(
                        label="batch_size", placeholder="1"))
                with gr.Row():
                    reg("ds_caption_dropout_rate", gr.Textbox(
                        label="caption_dropout_rate", placeholder="0.0"))
                    reg("ds_flip_aug", gr.Checkbox(value=False, label="flip_aug"))
                    reg("ds_random_crop", gr.Checkbox(value=False, label="random_crop"))
                    reg("ds_name", gr.Textbox(
                        label="dataset name (built TOML)", placeholder="my_char"))
                _tiers = [str(t) for t in server.list_target_res_tiers()]
                reg("target_res", gr.CheckboxGroup(
                    _tiers, value=[t for t in ("896", "1024") if t in _tiers],
                    label="Resolution tiers (constant-token; preprocess --target_res)"))
                reg("ds_tiers", gr.Textbox(
                    label="this subset's tiers (multi-scale; blank = all)",
                    placeholder="e.g. 1024 or 512,1024"))
                gr.Markdown(
                    "**Additional subsets** (optional) — one row per extra folder; "
                    "the fields above are subset #1. Leave empty for a single subset."
                )
                reg("ds_extra", gr.Dataframe(
                    headers=list(_DS_EXTRA_COLS),
                    datatype=["str", "str", "number", "number", "str",
                              "number", "bool", "bool", "str"],
                    type="array",
                    row_count=(0, "dynamic"),
                    label="Additional subsets (image_dir required per row)",
                ))
                reg_path(
                    "dataset_config",
                    file=True,
                    label="…or a dataset config TOML (overrides the fields above)",
                    placeholder="path/to/dataset.toml",
                )
                gr.Markdown(
                    "_Blank `image_dir` **and** `dataset_config` → the default "
                    "`base.toml` blueprint (`post_image_dataset/lora`). An explicit "
                    "`dataset_config` wins over the folder fields._"
                )
            # ── Auto-preprocess at train start ──────────────────────────────
            with gr.Accordion("Auto-preprocess at train start", open=False):
                gr.Markdown(
                    "Toggle **ON** and just hit **Start**: the subset folders above "
                    "(point `image_dir` at the **raw** images) are resized + cached "
                    "into `cache/<output_name>/` first, then training runs — one "
                    "subprocess. A completion marker skips it on the next run if "
                    "nothing changed. (Masking uses the Utils-tab SAM3/MIT toggles.)"
                )
                with gr.Row():
                    reg("auto_preprocess", gr.Checkbox(
                        value=False,
                        label="auto_preprocess (resize/cache then train)"))
                    reg("multiscale", gr.Checkbox(
                        value=False, label="multiscale (every tier, ≥2 tiers)"))
                    reg("drop_lowres", gr.Checkbox(
                        value=True, label="drop low-res (< 0.5MP)"))
                    reg("mask_enable", gr.Checkbox(
                        value=False, label="mask (SAM3 + MIT)"))
                with gr.Row():
                    reg("caption_shuffle_variants", gr.Textbox(
                        label="caption_shuffle_variants (caption variation)",
                        placeholder="4"))
                    reg("caption_tag_dropout_rate", gr.Textbox(
                        label="caption_tag_dropout_rate", placeholder="0.1"))

            # ── Sample prompts ──────────────────────────────────────────────
            with gr.Accordion("Sample images", open=True):
                reg_path(
                    "sample_prompts",
                    file=True,
                    label="Sample prompts file (--sample_prompts)",
                    placeholder="path/to/prompts.txt",
                )
                sample_editor = gr.Textbox(
                    label="…or edit prompts here and Save",
                    lines=4,
                    placeholder="a photo of sks dog --w 1024 --h 1024 --s 20",
                )
                save_samples_btn = gr.Button("Save prompts → file")

            # ── Advanced / monitor / run ────────────────────────────────────
            with gr.Accordion("Monitor & Run", open=True):
                with gr.Row():
                    reg(
                        "monitor",
                        gr.Checkbox(value=False, label="Web loss monitor (--monitor)"),
                    )
                with gr.Row():
                    reg(
                        "monitor_host",
                        gr.Textbox(value="127.0.0.1", label="Monitor host"),
                    )
                    reg(
                        "monitor_port",
                        gr.Textbox(label="Monitor port", placeholder="8766"),
                    )
                    reg(
                        "log_every_n_steps",
                        gr.Textbox(label="Log every N steps", placeholder="(default)"),
                    )

            # ── Training details (sd-scripts / LETS knobs → dedicated fields) ─
            with gr.Accordion("Training details (sd-scripts / LETS)", open=False):
                with gr.Row():
                    reg("mixed_precision", gr.Dropdown(
                        ["", "bf16", "fp16", "no"], value="", label="mixed_precision"))
                    reg("attn_mode", gr.Dropdown(
                        ["", "flash", "sdpa", "torch", "sageattn", "flex"],
                        value="", label="attn_mode"))
                    # torch_compile defaults ON (base.toml) → tri-state so the box can
                    # force it off too (blank = config default = on; the big speedup).
                    reg("torch_compile", gr.Dropdown(
                        ["", "on", "off"], value="",
                        label="torch_compile (blank = default on)"))
                    reg("save_precision", gr.Dropdown(
                        ["", "bf16", "fp16", "float"], value="", label="save_precision"))
                with gr.Row():
                    reg("loss_type", gr.Dropdown(
                        ["", "l2", "huber", "smooth_l1"], value="", label="loss_type"))
                    reg("huber_c", gr.Textbox(label="huber_c", placeholder="0.1"))
                    reg("huber_schedule", gr.Dropdown(
                        ["", "constant", "exponential", "snr"], value="",
                        label="huber_schedule"))
                with gr.Row():
                    reg("timestep_sampling", gr.Dropdown(
                        ["", "sigmoid", "uniform", "logit_normal", "shift"],
                        value="", label="timestep_sampling"))
                    reg("sigmoid_scale", gr.Textbox(
                        label="sigmoid_scale", placeholder="1.0"))
                    reg("weighting_scheme", gr.Dropdown(
                        ["", "logit_normal", "mode", "cosmap", "sigma_sqrt", "none"],
                        value="", label="weighting_scheme"))
                with gr.Row():
                    reg("logit_mean", gr.Textbox(label="logit_mean", placeholder="0.0"))
                    reg("logit_std", gr.Textbox(label="logit_std", placeholder="1.0"))
                    reg("t_min", gr.Textbox(label="t_min (σ 0–1)", placeholder="0.0"))
                    reg("t_max", gr.Textbox(label="t_max (σ 0–1)", placeholder="1.0"))
                with gr.Row():
                    reg("max_grad_norm", gr.Textbox(
                        label="max_grad_norm", placeholder="1.0"))
                    reg("gradient_accumulation_steps", gr.Textbox(
                        label="grad_accum_steps", placeholder="1"))
                    reg("blocks_to_swap", gr.Textbox(
                        label="blocks_to_swap", placeholder="0"))
                    reg("qwen3_max_token_length", gr.Textbox(
                        label="qwen3_max_token_length", placeholder="512"))
                with gr.Row():
                    reg("save_every_n_epochs", gr.Textbox(
                        label="save_every_n_epochs", placeholder="1"))
                with gr.Row():
                    reg("gradient_checkpointing", gr.Checkbox(
                        value=False, label="gradient_checkpointing"))
                    reg("network_train_unet_only", gr.Checkbox(
                        value=False, label="network_train_unet_only"))
                    reg("use_vae_cache", gr.Checkbox(
                        value=False, label="use_vae_cache (cache latents to disk)"))
                    reg("save_state", gr.Checkbox(value=False, label="save_state"))
                    reg("output_config", gr.Checkbox(
                        value=False, label="output_config (save_toml)"))
                # Disk-caching siblings of "cache latents to disk" + the caption-
                # variation technique (shuffled-caption-variant TE caches).
                with gr.Row():
                    reg("use_text_cache", gr.Checkbox(
                        value=False, label="use_text_cache (cache TE to disk)"))
                    reg("qwen_image_vae_2d", gr.Checkbox(
                        value=False,
                        label="qwen_image_vae_2d (~2x faster VAE caching)"))
                    reg("use_shuffled_caption_variants", gr.Checkbox(
                        value=False, label="use_shuffled_caption_variants"))
                    reg("use_shuffled_caption_variants_only", gr.Checkbox(
                        value=False, label="…_variants_only (skip pristine v0)"))
                reg_path("resume",
                    label="resume (saved training-state dir)",
                    placeholder="output/ckpt/<name>-state")
                gr.Markdown(
                    "*Blank/unchecked → defer to the `base→preset→method` config "
                    "chain. To force a bool **off**, use Extra CLI flags `--no-<flag>`. "
                    "Caption variants need TE caches built with "
                    "`caption_shuffle_variants > 0` at preprocess.*"
                )

            with gr.Accordion("Extra CLI flags", open=False):
                reg(
                    "extra_flags",
                    gr.Textbox(
                        label="Raw extra args appended verbatim",
                        placeholder="--no-masked_loss --network_weights path.safetensors",
                    ),
                )

            # ── Config file (load / save) — sd-scripts/LETS --config_file ───
            with gr.Accordion("Config file (load / save)", open=True):
                gr.Markdown(
                    "Load a LETS / kohya_ss / anima_lora ``--config_file`` TOML into "
                    "the form (key renames applied; unmapped keys fold into *Extra "
                    "CLI flags*), or save the current form as a runnable config."
                )
                with gr.Row():
                    config_path = gr.Textbox(
                        label="Config TOML path",
                        placeholder="configs/examples/lokr_came.toml",
                        scale=4,
                    )
                    cfg_browse_btn = gr.Button("📁", scale=0, min_width=46)
                    load_cfg_btn = gr.Button("Load → form", variant="secondary")
                    save_cfg_btn = gr.Button("Save form →", variant="secondary")
                config_status = gr.Markdown("")
                cfg_browse_btn.click(_pick_file, inputs=config_path, outputs=config_path)

            # ── Queue (saved runs, LoRA_Easy-style) ─────────────────────────
            with gr.Accordion("Queue (saved runs)", open=False):
                gr.Markdown(
                    "Stack runs to launch one at a time (single subprocess). "
                    "**Add** the current form, **Run next** launches the first queued "
                    "run; re-run after it finishes for the next."
                )
                queue_view = gr.JSON(label="Queued runs (id · name)")
                with gr.Row():
                    queue_add_btn = gr.Button("Add current → queue")
                    queue_run_btn = gr.Button("Run next", variant="primary")
                    queue_refresh_btn = gr.Button("Refresh")
                    queue_clear_btn = gr.Button("Clear queue", variant="stop")

        with gr.Tab("Utils"):
            gr.Markdown(
                "Utilities run as **direct subprocesses** (`tasks.py …`), like "
                "training — **mutually exclusive** with a training run (one at a "
                "time). Output streams to the training-log panel below; use **Stop** "
                "to cancel."
            )
            # ── Auto-batch search (tasks.py bench-autobatch) ────────────────
            with gr.Accordion("Auto-batch (max batch-size search)", open=True):
                _ab_tiers = [str(t) for t in server.list_target_res_tiers()]
                reg("ab_res", gr.CheckboxGroup(
                    _ab_tiers, value=[t for t in ("1024",) if t in _ab_tiers],
                    label="Resolutions to search (--res)"))
                with gr.Row():
                    reg("ab_max_batch", gr.Textbox(
                        label="max batch (--max-batch)", placeholder="8"))
                    reg("ab_optimizer_type", gr.Textbox(
                        label="optimizer_type", placeholder="AdamW"))
                    reg("ab_blocks_to_swap", gr.Textbox(
                        label="blocks_to_swap", placeholder="0"))
                    reg("ab_compile", gr.Checkbox(value=False, label="--compile"))
                with gr.Row():
                    reg("ab_network_module", gr.Textbox(
                        label="network_module", placeholder="networks.lora_anima"))
                    reg("ab_network_dim", gr.Textbox(
                        label="network_dim", placeholder="16"))
                    reg("ab_network_alpha", gr.Textbox(
                        label="network_alpha", placeholder="8"))
                with gr.Row():
                    reg("ab_network_args", gr.Textbox(
                        label="network_args (k=v …)", placeholder="algo=lokr factor=4"))
                ab_run_btn = gr.Button("Run auto-batch", variant="primary")
            # ── Masking (tasks.py mask: SAM3 + MIT) ─────────────────────────
            with gr.Accordion("Masking (SAM3 + MIT)", open=False):
                gr.Markdown(
                    "Masks the configured resized dir → `post_image_dataset/masks/` "
                    "(dirs from `preprocess.toml`/`base.toml`). SAM3 needs "
                    "`models/sam3/`; MIT needs `models/mit/model.pth`."
                )
                with gr.Row():
                    reg("mask_sam", gr.Checkbox(value=True, label="SAM3 (RUN_SAM_MASK)"))
                    reg("mask_mit", gr.Checkbox(value=True, label="MIT (RUN_MIT_MASK)"))
                    reg("mit_text_threshold", gr.Textbox(
                        label="MIT text threshold", placeholder="(default)"))
                    reg("mit_dilate", gr.Textbox(
                        label="MIT dilate", placeholder="(default)"))
                mask_run_btn = gr.Button("Run masking", variant="primary")

        # ── Actions (always visible below the tabs, kohya-style) ────────────
        with gr.Row():
            print_btn = gr.Button("Print training command", variant="secondary")
            start_btn = gr.Button("Start training", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop")
            status_btn = gr.Button("Refresh status", variant="secondary")

        out_cmd = gr.Code(label="train.py command", language="shell")
        out_status = gr.JSON(label="Result / status")

        # ── Live training log (the captured terminal output) ────────────────
        # train.py's console log is the sd-scripts RichHandler format. launch()
        # redirects the child subprocess's stdout/stderr to a logfile, which
        # server.log_tail() exposes. This panel mirrors the terminal live so
        # the GUI and the console stay linked.
        gr.Markdown("### Training log — terminal output (sd-scripts format)")
        with gr.Row():
            autorefresh = gr.Checkbox(value=True, label="Auto-refresh (2s)")
            refresh_log_btn = gr.Button("Refresh log now")
        out_log = gr.Textbox(
            label="stdout.log tail (the live console)",
            lines=22,
            max_lines=22,
            autoscroll=True,
            interactive=False,
        )
        log_timer = gr.Timer(2.0)

        # ── Handlers ────────────────────────────────────────────────────────
        def on_print(*vals):
            form = _collect(keys, vals)
            # Mirror launch(): subsets → a precached --dataset_config, so the preview
            # shows the same command Start would run. (launch rebuilds it itself.)
            if form.get("subsets") and not (form.get("dataset_config") or "").strip():
                try:
                    pc = server._build_precached_config(form)
                    if pc:
                        form["dataset_config"] = pc
                except Exception:  # noqa: BLE001, S110  (preview-only; ignore)
                    pass
            try:
                return " ".join(server.build_command(form))
            except Exception as exc:  # noqa: BLE001
                return f"# error building command: {exc}"

        def on_start(*vals):
            form = _collect(keys, vals)
            try:
                res = server.launch(form)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": str(exc)}
            cmd = res.get("command", "")
            return cmd, res

        def on_stop():
            return "", server.stop()

        def on_status():
            return server.status()

        def _run_util(fn, *vals):
            form = _collect(keys, vals)
            try:
                res = fn(form)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": str(exc)}
            return res.get("command", ""), res

        def on_autobatch(*vals):
            return _run_util(server.bench_autobatch, *vals)

        def on_masking(*vals):
            return _run_util(server.run_masking, *vals)

        def _queue_brief():
            # Show only id + name (the full per-run form is large and noisy).
            return [{"id": i.get("id"), "name": i.get("name")}
                    for i in server.queue_list()]

        def on_queue_add(*vals):
            form = _collect(keys, vals)
            server.queue_add((form.get("output_name") or "run").strip() or "run", form)
            return _queue_brief()

        def on_queue_run():
            res = server.queue_run()
            return _queue_brief(), res

        def on_queue_refresh():
            return _queue_brief()

        def on_queue_clear():
            server.queue_clear()
            return _queue_brief()

        def on_save_samples(name, text):
            res = server.save_sample_prompts(name or "sample", text or "")
            # Push the written path into the sample_prompts field on success.
            return res.get("path", "") if res.get("ok") else ""

        def on_log_tick():
            res = server.log_tail(120)
            lines = res.get("lines") or []
            if not lines and res.get("note"):
                return res["note"]
            return "\n".join(lines)

        def on_load_config(path):
            """Read a config TOML → push values into the matching form fields."""
            p = (path or "").strip()
            if not p:
                return [gr.update() for _ in keys] + ["⚠ enter a config file path"]
            try:
                with open(p, encoding="utf-8") as fh:
                    form = load_toml_to_form(fh.read())
            except Exception as exc:  # noqa: BLE001
                return [gr.update() for _ in keys] + [f"❌ load error: {exc}"]
            updates = [
                gr.update(value=form[k]) if k in form else gr.update() for k in keys
            ]
            note = (
                f"✓ loaded {len(form)} field(s) from `{p}`"
                + (" — unmapped keys are in *Extra CLI flags*"
                   if form.get("extra_flags") else "")
            )
            return updates + [note]

        def on_save_config(path, *vals):
            """Current form → a runnable --config_file TOML on disk."""
            import os

            form = _collect(keys, vals)
            p = (path or "").strip() or "output/gui_config.toml"
            try:
                text = save_form_to_toml(form)
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(text)
            except Exception as exc:  # noqa: BLE001
                return f"❌ save error: {exc}"
            return f"✓ saved → `{p}` (run: `python train.py --config_file {p}`)"

        load_cfg_btn.click(
            on_load_config, inputs=config_path, outputs=inputs + [config_status]
        )
        save_cfg_btn.click(
            on_save_config, inputs=[config_path, *inputs], outputs=config_status
        )

        print_btn.click(on_print, inputs=inputs, outputs=out_cmd)
        start_btn.click(on_start, inputs=inputs, outputs=[out_cmd, out_status])
        stop_btn.click(on_stop, inputs=None, outputs=[out_cmd, out_status])
        status_btn.click(on_status, inputs=None, outputs=out_status)
        ab_run_btn.click(on_autobatch, inputs=inputs, outputs=[out_cmd, out_status])
        mask_run_btn.click(on_masking, inputs=inputs, outputs=[out_cmd, out_status])
        queue_add_btn.click(on_queue_add, inputs=inputs, outputs=queue_view)
        queue_run_btn.click(on_queue_run, inputs=None, outputs=[queue_view, out_status])
        queue_refresh_btn.click(on_queue_refresh, inputs=None, outputs=queue_view)
        queue_clear_btn.click(on_queue_clear, inputs=None, outputs=queue_view)
        # Live log: tick every 2s while Auto-refresh is on; the checkbox toggles
        # the timer so the poll stops when the panel isn't being watched.
        log_timer.tick(on_log_tick, inputs=None, outputs=out_log)
        refresh_log_btn.click(on_log_tick, inputs=None, outputs=out_log)
        autorefresh.change(
            lambda on: gr.Timer(active=bool(on)), inputs=autorefresh, outputs=log_timer
        )
        # `sample_prompts` is registered; locate its component to receive the path.
        sample_path_comp = inputs[keys.index("sample_prompts")]
        save_samples_btn.click(
            on_save_samples,
            inputs=[inputs[keys.index("output_name")], sample_editor],
            outputs=sample_path_comp,
        )

        # ── Conflict / dependency greying (ported from the web GUI's
        #    CONFLICT_RULES + ARG_DEPS) — disable (and reset) a field when an active
        #    option makes it a no-op or incompatible. Every relevant change (and the
        #    initial load) recomputes all targets from the driver values. ──────────
        _dep_drivers = ["use_vae_cache", "use_text_cache", "use_constantcosine",
                        "loss_type", "timestep_sampling", "weighting_scheme"]
        _dep_targets = ["ds_random_crop", "ds_caption_dropout_rate",
                        "lr_scheduler_type", "huber_c", "huber_schedule",
                        "sigmoid_scale", "logit_mean", "logit_std"]

        def _gray(on: bool, reset=None):
            if on:
                return gr.update(interactive=True)
            if reset is not None:
                return gr.update(interactive=False, value=reset)
            return gr.update(interactive=False)

        def _recompute_deps(use_vae, use_text, use_cc, loss, ts, ws):
            huber = loss in ("huber", "smooth_l1")
            return [
                _gray(not use_vae, reset=False),   # random_crop ↮ cached latents
                _gray(not use_text, reset="0"),    # caption_dropout ↮ cached TE
                _gray(not use_cc),                 # lr_scheduler ← overridden by cc
                _gray(huber),                      # huber_c only for huber/smooth_l1
                _gray(huber),                      # huber_schedule
                _gray(ts in ("", "sigmoid")),      # sigmoid_scale: sigmoid family
                _gray(ws == "logit_normal"),       # logit_mean
                _gray(ws == "logit_normal"),       # logit_std
            ]

        _dep_in = [by_key[k] for k in _dep_drivers]
        _dep_out = [by_key[k] for k in _dep_targets]
        for _drv in _dep_in:
            _drv.change(_recompute_deps, _dep_in, _dep_out)
        demo.load(_recompute_deps, _dep_in, _dep_out)

    # Stash the field order on the app for any callers / debugging.
    FIELD_KEYS[:] = keys
    return demo


def serve(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    """Launch the Gradio server (blocking)."""
    import inspect
    import socket

    # The requested port is commonly taken — 7860 is ALSO gradio/forge-neo's
    # default, or a prior GUI is still up. Gradio only tries the single port we
    # pass and hard-fails ("Cannot find empty port in range: 7860-7860"), so scan
    # upward for the first free one ourselves (mirrors the old web GUI).
    def _free_port(start: int, span: int = 20) -> int:
        for p in range(start, start + span):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, p))
                    return p
                except OSError:
                    continue
        return start  # none free in range — let gradio raise its own message

    bound = _free_port(port)
    if bound != port:
        print(f"\n  port {port} is busy (forge-neo/another app) — using {bound}\n")

    demo = build_app(default_port=bound)
    kwargs = {
        "server_name": host,
        "server_port": bound,
        "inbrowser": open_browser,
        "show_api": False,  # hide the auto-generated API page
    }
    # Tolerate gradio version drift: a launch() whose signature dropped/renamed a
    # kwarg (e.g. show_api on some 5.x/6.x builds) would otherwise TypeError and
    # kill the GUI. If launch() has no **kwargs catch-all, pass only what it
    # actually accepts.
    try:
        params = inspect.signature(demo.launch).parameters
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if not has_var_kw:
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    demo.launch(**kwargs)
