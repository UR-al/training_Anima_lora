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


# Free-arg groups rendered as name → value ROWS (the LoRA_Easy "ADD NETWORK ARG"
# widget). The GUI shows a fixed pool of rows per group (keys "{group}__k{i}" /
# "{group}__v{i}"); _collect folds the non-blank rows into the space-joined
# `key=value …` string the backend reads under the group key, and config-load splits
# the inline string back into the rows. group → number of rows in the pool.
_ARG_ROW_GROUPS = {
    "network_args": 6,
    "optimizer_args": 5,
    "lr_scheduler_args": 4,
    "ab_network_args": 4,
}


def _arg_rows_to_inline(form: dict, group: str, n: int) -> str:
    """Collapse a group's name/value rows into ``k1=v1 k2=v2 …`` and POP the row keys
    from ``form``. A value with spaces is auto-quoted so it survives the backend's
    _arg_split; a name with no value emits the bare key (flag-style)."""
    toks: list[str] = []
    for i in range(1, n + 1):
        k = str(form.pop(f"{group}__k{i}", "") or "").strip()
        v = str(form.pop(f"{group}__v{i}", "") or "").strip()
        if not k:
            continue
        if v == "":
            toks.append(k)
        else:
            if " " in v and not (v[:1] in "\"'" and v[-1:] == v[:1]):
                v = f'"{v}"'
            toks.append(f"{k}={v}")
    return " ".join(toks)


def _inline_to_arg_rows(text, n: int) -> list:
    """Inverse for config-load: split a space-joined ``key=value`` string into n
    (name, value) pairs (quote-aware), padded/truncated to exactly n rows."""
    import shlex

    s = str(text or "").strip()
    rows: list = []
    if s:
        try:
            toks = shlex.split(s)
        except ValueError:
            toks = s.split()
        for t in toks:
            k, _, v = t.partition("=")
            rows.append((k, v))
    return rows[:n] + [("", "")] * max(0, n - len(rows))


_MISSING = object()


def _interactive_states(form: dict) -> dict:
    """field key → {interactive: bool[, value: reset]} for the dep-greying targets,
    computed server-side from a just-loaded form. on_load_config folds these into its
    OWN output so a config load fires NO secondary .change/.then event — which used to
    race the per-driver cascade and wedge huber_c/sigmoid_scale/… on a spinner."""
    loss = str(form.get("loss_type", "") or "")
    ts = str(form.get("timestep_sampling", "") or "")
    ws = str(form.get("weighting_scheme", "") or "")
    use_vae = bool(form.get("use_vae_cache"))
    use_text = bool(form.get("use_text_cache"))
    use_cc = bool(form.get("use_constantcosine"))
    huber = loss in ("huber", "smooth_l1")

    def g(on, reset=_MISSING):
        if on:
            return {"interactive": True}
        if reset is _MISSING:
            return {"interactive": False}
        return {"interactive": False, "value": reset}

    return {
        "ds_random_crop": g(use_vae, reset=False),
        "ds_caption_dropout_rate": g(use_text, reset="0"),
        "lr_scheduler_type": g(not use_cc),
        "huber_c": g(huber),
        "huber_schedule": g(huber),
        "sigmoid_scale": g(ts in ("", "sigmoid")),
        "logit_mean": g(ws == "logit_normal"),
        "logit_std": g(ws == "logit_normal"),
    }


def _collect(keys: list[str], values) -> dict:
    """Zip positional handler args back into the backend `form` dict."""
    form = dict(zip(keys, values))
    # Gradio Textbox yields "" for empty; backend treats "" as "use default".
    # Coerce checkbox-style truthiness through untouched (already bool).
    for group, n in _ARG_ROW_GROUPS.items():  # name/value rows → inline key=value
        inline = _arg_rows_to_inline(form, group, n)  # also pops the row keys
        if inline:
            form[group] = inline
    _assemble_dataset(form)
    return form


# Per-subset fields. The GUI renders N fixed subset BLOCKS — the primary uses the
# "ds_" prefix, extras "ds2_"/"ds3_"/… — and each block's image_dir gates whether it
# becomes a subset. (suffix → subset-dict key, caster) + bool suffixes. The backend
# consumes the assembled form["subsets"]; cache_dir is primary-only and shared to all
# subsets (so extras have no cache_dir field). Replaces the old single-primary + one
# gr.Dataframe grid (which, under Gradio 6.x, could only ever add a single extra row).
N_SUBSETS = 4  # primary + 3 extras; for more, point the dataset_config field at a TOML
_DS_PREFIXES = ("ds_",) + tuple(f"ds{i}_" for i in range(2, N_SUBSETS + 1))
_DS_FIELD_SUFFIXES = (
    ("num_repeats", "num_repeats", int),
    ("keep_tokens", "keep_tokens", int),
    ("caption_extension", "caption_extension", str),
    ("caption_dropout_rate", "caption_dropout_rate", float),
    ("batch_size", "batch_size", int),
)
_DS_BOOL_SUFFIXES = (
    ("flip_aug", "flip_aug"),
    ("random_crop", "random_crop"),
    ("gradient_checkpointing", "gradient_checkpointing"),
)
# Flat per-block keys stripped from the form after assembling form["subsets"], so the
# backend's auto-arg catch-all never misreads them as train.py flags. ds_name and
# target_res stay (the backend reads them directly).
_DS_POP_KEYS = ("ds_cache_dir",) + tuple(
    f"{p}{suf}"
    for p in _DS_PREFIXES
    for suf in (
        "image_dir",
        "tiers",
        *(s for s, _k, _c in _DS_FIELD_SUFFIXES),
        *(s for s, _k in _DS_BOOL_SUFFIXES),
    )
)


def _tier_list(text) -> list[int]:
    """Parse a free-form tier string (\"512,1024\" / \"1024\") into ints."""
    import re

    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


def _subset_from_prefix(form: dict, prefix: str):
    """Build one subset dict from a block's prefixed form fields, or None when the
    block has no image_dir (an unused block)."""
    img = str(form.get(f"{prefix}image_dir") or "").strip()
    if not img:
        return None
    sub: dict = {"image_dir": img}
    for suf, sk, cast in _DS_FIELD_SUFFIXES:
        v = form.get(f"{prefix}{suf}")
        if v in (None, ""):
            continue
        try:
            sub[sk] = cast(v)
        except (TypeError, ValueError):
            pass
    for suf, sk in _DS_BOOL_SUFFIXES:
        if form.get(f"{prefix}{suf}"):
            sub[sk] = True
    tiers = _tier_list(form.get(f"{prefix}tiers"))
    if tiers:
        sub["tiers"] = tiers
    return sub


def _assemble_dataset(form: dict) -> None:
    """Fold the N per-subset blocks into ``form['subsets']`` (primary first). The
    primary subset's cache_dir is shared to every subset (extras have no cache_dir
    field). No-op when no block has an image_dir (defer to the base.toml blueprint /
    an explicit --dataset_config). Mutates ``form``."""
    primary_cache = str(form.get("ds_cache_dir") or "").strip()
    subs: list[dict] = []
    for prefix in _DS_PREFIXES:
        sub = _subset_from_prefix(form, prefix)
        if sub is None:
            continue
        if primary_cache:  # all subsets share the primary subset's cache_dir
            sub["cache_dir"] = primary_cache
        subs.append(sub)
    if subs and not form.get("subsets"):
        form["subsets"] = subs
    for k in _DS_POP_KEYS:  # don't leak the flat block helpers to the backend form
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
        initdir = (
            current
            if os.path.isdir(current)
            else (os.path.dirname(current) or os.getcwd())
        )
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
        # show_progress="hidden": the native dialog blocks the handler, so without this
        # Gradio paints a "processing" spinner over the field for the whole time the
        # picker is open. Hiding it keeps the field looking idle while you browse.
        btn.click(
            _pick_file if file else _pick_dir,
            inputs=tb,
            outputs=tb,
            show_progress="hidden",
        )
        return tb

    def reg_arg_rows(group, *, title):
        """LoRA_Easy-style "ADD NETWORK ARG" editor: a header + a fixed pool of
        name → value rows (Enter Arg Name | Enter Arg Value | 🗑 clear). Each row
        reg()s as ``{group}__k{i}`` / ``{group}__v{i}``; _collect folds the non-blank
        rows into the inline ``key=value`` string the backend reads under ``group``."""
        n = _ARG_ROW_GROUPS[group]
        gr.Markdown(f"**{title}** — one key → value per row (blank rows ignored)")
        for i in range(1, n + 1):
            with gr.Row():
                ktb = gr.Textbox(
                    placeholder="Enter Arg Name",
                    show_label=False,
                    container=False,
                    scale=5,
                )
                vtb = gr.Textbox(
                    placeholder="Enter Arg Value",
                    show_label=False,
                    container=False,
                    scale=5,
                )
                clr = gr.Button("🗑", scale=0, min_width=44)
            reg(f"{group}__k{i}", ktb)
            reg(f"{group}__v{i}", vtb)
            # outputs bind THIS row's components (evaluated at click-registration);
            # the lambda returns constants, so there's no loop-closure capture bug.
            clr.click(lambda: ("", ""), outputs=[ktb, vtb], show_progress="hidden")

    with gr.Blocks(title="Anima LoRA Trainer") as demo:
        gr.Markdown(
            "# Anima LoRA Trainer\n"
            "Gradio front-end (kohya-style layout) driving **this repo's** "
            "`train.py --method <name> --preset <name>`. Blank fields defer to the "
            "`base.toml → preset → method` config chain. Start runs train.py "
            "directly; its log streams below and to the terminal."
        )

        with gr.Tab("LoRA"):
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
                cfg_browse_btn.click(
                    _pick_file,
                    inputs=config_path,
                    outputs=config_path,
                    show_progress="hidden",
                )

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

            # ── Dataset (subsets) ───────────────────────────────────────────
            with gr.Accordion("Dataset (subsets)", open=True):
                gr.Markdown(
                    f"Define up to **{N_SUBSETS}** subsets as folder blocks (old-GUI / "
                    "LoRA_Easy style). **Subset #1** is the primary; the rest are "
                    "optional (blank `image_dir` = unused). Run `make preprocess` "
                    "first (point `image_dir` at the resized+cached dir), or toggle "
                    "**Auto-preprocess** below to resize/cache raw folders on Start. "
                    "`cache_dir` is set once on the primary and **shared by every "
                    f"subset**. For more than {N_SUBSETS} subsets, use a "
                    "`--dataset_config` TOML."
                )
                with gr.Row():
                    reg(
                        "ds_name",
                        gr.Textbox(
                            label="dataset name (built TOML)", placeholder="my_char"
                        ),
                    )
                _tiers = [str(t) for t in server.list_target_res_tiers()]
                reg(
                    "target_res",
                    gr.CheckboxGroup(
                        _tiers,
                        value=[t for t in ("896", "1024") if t in _tiers],
                        label="Resolution tiers (constant-token; preprocess --target_res)",
                    ),
                )

                def _subset_block(idx: int) -> None:
                    """Register one subset block's fields. Primary (idx==1) carries the
                    shared cache_dir; extras don't. reg() keeps keys+inputs in lockstep,
                    so ds{idx}_* round-trip through _collect / on_load_config unchanged."""
                    prefix = "ds_" if idx == 1 else f"ds{idx}_"
                    primary = idx == 1
                    title = (
                        "Subset #1 (primary)"
                        if primary
                        else f"Subset #{idx} (optional)"
                    )
                    with gr.Accordion(title, open=primary):
                        if primary:
                            with gr.Row():
                                reg_path(
                                    f"{prefix}image_dir",
                                    label="image_dir (resized + cached images)",
                                    placeholder="post_image_dataset/resized",
                                )
                                reg_path(
                                    "ds_cache_dir",
                                    label="cache_dir (shared by all subsets; blank = default)",
                                    placeholder="post_image_dataset/lora",
                                )
                        else:
                            reg_path(
                                f"{prefix}image_dir",
                                label="image_dir (blank = unused subset)",
                                placeholder="this subset's resized+cached images",
                            )
                        with gr.Row():
                            reg(
                                f"{prefix}num_repeats",
                                gr.Textbox(label="num_repeats", placeholder="1"),
                            )
                            reg(
                                f"{prefix}keep_tokens",
                                gr.Textbox(label="keep_tokens", placeholder="0"),
                            )
                            reg(
                                f"{prefix}caption_extension",
                                gr.Textbox(
                                    label="caption_extension", placeholder=".txt"
                                ),
                            )
                            reg(
                                f"{prefix}batch_size",
                                gr.Textbox(
                                    label="batch_size", placeholder="(dataset default)"
                                ),
                            )
                        with gr.Row():
                            reg(
                                f"{prefix}caption_dropout_rate",
                                gr.Textbox(
                                    label="caption_dropout_rate", placeholder="0.0"
                                ),
                            )
                            reg(
                                f"{prefix}tiers",
                                gr.Textbox(
                                    label="this subset's tiers (blank = all)",
                                    placeholder="e.g. 1024 or 512,1024",
                                ),
                            )
                        with gr.Row():
                            reg(
                                f"{prefix}flip_aug",
                                gr.Checkbox(value=False, label="flip_aug"),
                            )
                            reg(
                                f"{prefix}random_crop",
                                gr.Checkbox(value=False, label="random_crop"),
                            )
                            reg(
                                f"{prefix}gradient_checkpointing",
                                gr.Checkbox(
                                    value=False,
                                    label="gradient_checkpointing (this subset's tier)",
                                ),
                            )

                for _i in range(1, N_SUBSETS + 1):
                    _subset_block(_i)

                reg_path(
                    "dataset_config",
                    file=True,
                    label="…or a dataset config TOML (overrides the blocks above)",
                    placeholder="path/to/dataset.toml",
                )
                gr.Markdown(
                    "_Blank every `image_dir` **and** `dataset_config` → the default "
                    "`base.toml` blueprint (`post_image_dataset/lora`). An explicit "
                    "`dataset_config` wins over the blocks above._"
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
                    reg(
                        "auto_preprocess",
                        gr.Checkbox(
                            value=False,
                            label="auto_preprocess (resize/cache then train)",
                        ),
                    )
                    reg(
                        "multiscale",
                        gr.Checkbox(
                            value=False, label="multiscale (every tier, ≥2 tiers)"
                        ),
                    )
                    reg(
                        "drop_lowres",
                        gr.Checkbox(value=True, label="drop low-res (< 0.5MP)"),
                    )
                    reg(
                        "mask_enable",
                        gr.Checkbox(value=False, label="mask (SAM3 + MIT)"),
                    )
                with gr.Row():
                    reg(
                        "caption_shuffle_variants",
                        gr.Textbox(
                            label="caption_shuffle_variants (caption variation)",
                            placeholder="4",
                        ),
                    )
                    reg(
                        "caption_tag_dropout_rate",
                        gr.Textbox(label="caption_tag_dropout_rate", placeholder="0.1"),
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
                reg_arg_rows("network_args", title="Network args")

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
                # Per-optimizer / per-scheduler arg help (backend.optimizer_arg_help):
                # selecting either dropdown lists its accepted args + plain-language desc.
                opt_help = gr.Markdown(
                    "_Select an optimizer or scheduler to see its arguments._"
                )

                def _fmt_opt_help(name):
                    r = server.optimizer_arg_help((name or "").strip())
                    if not r.get("ok"):
                        return r.get("note") or f"_no help for `{name}`_"
                    head = r.get("cls") or name
                    lines = [f"**{head}**"]
                    if r.get("note"):
                        lines.append(r["note"])
                    for a in r.get("args", []):
                        dv = a.get("default")
                        dv = "" if dv is None else f" = `{dv}`"
                        req = " **(required)**" if a.get("required") else ""
                        lines.append(f"- `{a['name']}`{dv}{req} — {a.get('desc', '')}")
                    return "\n".join(lines)

                by_key["optimizer_type"].change(
                    _fmt_opt_help, by_key["optimizer_type"], opt_help
                )
                by_key["lr_scheduler_type"].change(
                    _fmt_opt_help, by_key["lr_scheduler_type"], opt_help
                )
                with gr.Row():
                    reg(
                        "lr_warmup_steps",
                        gr.Textbox(label="LR warmup steps", placeholder="(default)"),
                    )
                reg_arg_rows("optimizer_args", title="Optimizer args")
                reg_arg_rows("lr_scheduler_args", title="LR scheduler args")
                # constant→cosine one-shot: hold constant LR for the planned run,
                # then extend with N cosine-decay epochs (LR→floor) in the SAME run.
                # Overrides lr_scheduler (which greys out when this is on).
                with gr.Row():
                    reg(
                        "use_constantcosine",
                        gr.Checkbox(
                            value=False, label="use_constantcosine (constant→cosine)"
                        ),
                    )
                    reg(
                        "constantcosine_tail_epochs",
                        gr.Textbox(
                            label="constantcosine_tail_epochs", placeholder="0 = off"
                        ),
                    )
                    reg(
                        "lr_scheduler_min_lr_ratio",
                        gr.Textbox(
                            label="min_lr_ratio (cosine floor)", placeholder="0.0"
                        ),
                    )

            # ── LoRA / LR extras (kohya-parity promotions) ──────────────────
            with gr.Accordion("LoRA / LR extras", open=False):
                with gr.Row():
                    reg(
                        "train_batch_size",
                        gr.Textbox(
                            label="train_batch_size (global)",
                            placeholder="(dataset/subset default)",
                        ),
                    )
                    reg(
                        "max_train_steps",
                        gr.Textbox(
                            label="max_train_steps", placeholder="(epochs-driven)"
                        ),
                    )
                with gr.Row():
                    reg_path(
                        "network_weights",
                        file=True,
                        label="network_weights (warm-start adapter)",
                        placeholder="path/to/existing.safetensors",
                    )
                    reg(
                        "dim_from_weights",
                        gr.Checkbox(value=False, label="dim_from_weights (infer rank)"),
                    )
                with gr.Row():
                    reg(
                        "unet_lr",
                        gr.Textbox(
                            label="unet_lr (DiT adapter LR)", placeholder="(=lr)"
                        ),
                    )
                    reg(
                        "llm_adapter_lr",
                        gr.Textbox(
                            label="llm_adapter_lr (Qwen3→DiT)",
                            placeholder="(blank=off)",
                        ),
                    )
                    reg(
                        "text_encoder_lr",
                        gr.Textbox(
                            label="text_encoder_lr (only w/ train_llm_adapter)",
                            placeholder="(frozen by default)",
                        ),
                    )
                with gr.Row():
                    reg(
                        "scale_weight_norms",
                        gr.Textbox(
                            label="scale_weight_norms (max-norm; 1.0 typical)",
                            placeholder="(off)",
                        ),
                    )
                    reg(
                        "network_dropout",
                        gr.Textbox(label="network_dropout", placeholder="(off)"),
                    )
                with gr.Row():
                    reg(
                        "lr_scheduler_num_cycles",
                        gr.Textbox(
                            label="lr_scheduler_num_cycles (cosine restarts)",
                            placeholder="1",
                        ),
                    )
                    reg(
                        "lr_scheduler_power",
                        gr.Textbox(
                            label="lr_scheduler_power (polynomial)", placeholder="1"
                        ),
                    )

            # ── Saving & checkpoints ────────────────────────────────────────
            with gr.Accordion("Saving & checkpoints", open=False):
                with gr.Row():
                    reg(
                        "save_every_n_steps",
                        gr.Textbox(label="Save every N steps", placeholder="(off)"),
                    )
                    reg(
                        "save_last_n_steps",
                        gr.Textbox(label="Save last N steps", placeholder="(keep all)"),
                    )
                    reg(
                        "save_last_n_steps_state",
                        gr.Textbox(
                            label="Save last N steps state", placeholder="(keep all)"
                        ),
                    )
                with gr.Row():
                    reg(
                        "save_last_n_epochs",
                        gr.Textbox(
                            label="Save last N epochs", placeholder="(keep all)"
                        ),
                    )
                    reg(
                        "save_last_n_epochs_state",
                        gr.Textbox(
                            label="Save last N epochs state", placeholder="(keep all)"
                        ),
                    )
                    reg(
                        "save_state_on_train_end",
                        gr.Checkbox(value=False, label="save_state_on_train_end"),
                    )

            # ── Performance / memory / caching ──────────────────────────────
            with gr.Accordion("Performance / memory / caching", open=False):
                with gr.Row():
                    reg(
                        "highvram",
                        gr.Checkbox(value=False, label="highvram (keep more in VRAM)"),
                    )
                    reg(
                        "lowram",
                        gr.Checkbox(value=False, label="lowram (models→VRAM not RAM)"),
                    )
                    reg(
                        "persistent_data_loader_workers",
                        gr.Checkbox(
                            value=False, label="persistent_data_loader_workers"
                        ),
                    )
                with gr.Row():
                    reg(
                        "max_data_loader_n_workers",
                        gr.Textbox(
                            label="max_data_loader_n_workers", placeholder="(default)"
                        ),
                    )
                    reg(
                        "vae_batch_size",
                        gr.Textbox(
                            label="vae_batch_size (caching)", placeholder="(default)"
                        ),
                    )
                with gr.Row():
                    reg(
                        "masked_loss",
                        gr.Dropdown(
                            ["", "on", "off"],
                            value="",
                            label="masked_loss (blank = default on)",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "skip_cache_check",
                        gr.Dropdown(
                            ["", "on", "off"],
                            value="",
                            label="skip_cache_check (blank = default on)",
                            allow_custom_value=True,
                        ),
                    )
                with gr.Row():
                    reg(
                        "save_model_as",
                        gr.Dropdown(
                            ["", "safetensors", "ckpt", "pt"],
                            value="",
                            label="save_model_as (blank = safetensors)",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "t5_max_token_length",
                        gr.Textbox(label="t5_max_token_length", placeholder="512"),
                    )
                    reg(
                        "vae_disable_cache",
                        gr.Checkbox(
                            value=False,
                            label="vae_disable_cache (disable VAE internal tiling cache)",
                        ),
                    )

            # ── Logging & metadata ──────────────────────────────────────────
            with gr.Accordion("Logging & metadata", open=False):
                gr.Markdown(
                    "TensorBoard logging dir is auto-set to `<output_dir>/log`. "
                    "Pick a tracker + (for W&B) a run name / API key."
                )
                with gr.Row():
                    reg(
                        "log_with",
                        gr.Dropdown(
                            ["", "tensorboard", "wandb", "all"],
                            value="",
                            label="log_with",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "wandb_run_name",
                        gr.Textbox(label="wandb_run_name", placeholder="(optional)"),
                    )
                    reg(
                        "log_tracker_name",
                        gr.Textbox(label="log_tracker_name", placeholder="(optional)"),
                    )
                reg(
                    "wandb_api_key",
                    gr.Textbox(
                        label="wandb_api_key", placeholder="(optional)", type="password"
                    ),
                )
                reg(
                    "training_comment",
                    gr.Textbox(
                        label="training_comment (stored in checkpoint metadata)",
                        placeholder="(optional)",
                    ),
                )

            # ── Advanced training details (sd-scripts / LETS knobs) ──────────
            with gr.Accordion("Training details (sd-scripts / LETS)", open=False):
                with gr.Row():
                    reg(
                        "mixed_precision",
                        gr.Dropdown(
                            ["", "bf16", "fp16", "no"],
                            value="",
                            label="mixed_precision",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "attn_mode",
                        gr.Dropdown(
                            ["", "flash", "sdpa", "torch", "sageattn", "flex"],
                            value="",
                            label="attn_mode",
                            allow_custom_value=True,
                        ),
                    )
                    # torch_compile defaults ON (base.toml) → tri-state so the box can
                    # force it off too (blank = config default = on; the big speedup).
                    reg(
                        "torch_compile",
                        gr.Dropdown(
                            ["", "on", "off"],
                            value="",
                            label="torch_compile (blank = default on)",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "save_precision",
                        gr.Dropdown(
                            ["", "bf16", "fp16", "float"],
                            value="",
                            label="save_precision",
                            allow_custom_value=True,
                        ),
                    )
                with gr.Row():
                    reg(
                        "loss_type",
                        gr.Dropdown(
                            ["", "l2", "huber", "smooth_l1"],
                            value="",
                            label="loss_type",
                            allow_custom_value=True,
                        ),
                    )
                    reg("huber_c", gr.Textbox(label="huber_c", placeholder="0.1"))
                    reg(
                        "huber_schedule",
                        gr.Dropdown(
                            ["", "constant", "exponential", "snr"],
                            value="",
                            label="huber_schedule",
                            allow_custom_value=True,
                        ),
                    )
                with gr.Row():
                    reg(
                        "timestep_sampling",
                        gr.Dropdown(
                            ["", "sigmoid", "uniform", "logit_normal", "shift"],
                            value="",
                            label="timestep_sampling",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "sigmoid_scale",
                        gr.Textbox(label="sigmoid_scale", placeholder="1.0"),
                    )
                    reg(
                        "weighting_scheme",
                        gr.Dropdown(
                            [
                                "",
                                "logit_normal",
                                "mode",
                                "cosmap",
                                "sigma_sqrt",
                                "none",
                            ],
                            value="",
                            label="weighting_scheme",
                            allow_custom_value=True,
                        ),
                    )
                with gr.Row():
                    reg("logit_mean", gr.Textbox(label="logit_mean", placeholder="0.0"))
                    reg("logit_std", gr.Textbox(label="logit_std", placeholder="1.0"))
                    reg("t_min", gr.Textbox(label="t_min (σ 0–1)", placeholder="0.0"))
                    reg("t_max", gr.Textbox(label="t_max (σ 0–1)", placeholder="1.0"))
                with gr.Row():
                    reg(
                        "max_grad_norm",
                        gr.Textbox(label="max_grad_norm", placeholder="1.0"),
                    )
                    reg(
                        "gradient_accumulation_steps",
                        gr.Textbox(label="grad_accum_steps", placeholder="1"),
                    )
                    reg(
                        "blocks_to_swap",
                        gr.Textbox(label="blocks_to_swap", placeholder="0"),
                    )
                    reg(
                        "qwen3_max_token_length",
                        gr.Textbox(label="qwen3_max_token_length", placeholder="512"),
                    )
                with gr.Row():
                    reg(
                        "save_every_n_epochs",
                        gr.Textbox(label="save_every_n_epochs", placeholder="1"),
                    )
                with gr.Row():
                    reg(
                        "gradient_checkpointing",
                        gr.Checkbox(value=False, label="gradient_checkpointing"),
                    )
                    reg(
                        "network_train_unet_only",
                        gr.Checkbox(value=False, label="network_train_unet_only"),
                    )
                    reg(
                        "use_vae_cache",
                        gr.Checkbox(
                            value=False, label="use_vae_cache (cache latents to disk)"
                        ),
                    )
                    reg("save_state", gr.Checkbox(value=False, label="save_state"))
                    reg(
                        "output_config",
                        gr.Checkbox(value=False, label="output_config (save_toml)"),
                    )
                # Disk-caching siblings of "cache latents to disk" + the caption-
                # variation technique (shuffled-caption-variant TE caches).
                with gr.Row():
                    reg(
                        "use_text_cache",
                        gr.Checkbox(
                            value=False, label="use_text_cache (cache TE to disk)"
                        ),
                    )
                    reg(
                        "qwen_image_vae_2d",
                        gr.Checkbox(
                            value=False,
                            label="qwen_image_vae_2d (~2x faster VAE caching)",
                        ),
                    )
                    reg(
                        "use_shuffled_caption_variants",
                        gr.Checkbox(value=False, label="use_shuffled_caption_variants"),
                    )
                    reg(
                        "use_shuffled_caption_variants_only",
                        gr.Checkbox(
                            value=False, label="…_variants_only (skip pristine v0)"
                        ),
                    )
                reg_path(
                    "resume",
                    label="resume (saved training-state dir)",
                    placeholder="output/ckpt/<name>-state",
                )
                gr.Markdown(
                    "*Blank/unchecked → defer to the `base→preset→method` config "
                    "chain. To force a bool **off**, use Extra CLI flags `--no-<flag>`. "
                    "Caption variants need TE caches built with "
                    "`caption_shuffle_variants > 0` at preprocess.*"
                )

            # ── Sample images ───────────────────────────────────────────────
            with gr.Accordion("Sample images", open=True):
                with gr.Row():
                    reg(
                        "sample_every_n_steps",
                        gr.Textbox(label="Sample every N steps", placeholder="(off)"),
                    )
                    reg(
                        "sample_every_n_epochs",
                        gr.Textbox(label="Sample every N epochs", placeholder="(off)"),
                    )
                with gr.Row():
                    reg(
                        "sample_sampler",
                        gr.Dropdown(
                            ["euler", "er_sde", "lcm"],
                            value="euler",
                            label="Sample sampler (--sample_sampler)",
                            allow_custom_value=True,
                        ),
                    )
                    reg(
                        "sample_at_first",
                        gr.Checkbox(
                            value=False, label="Sample at first (before training)"
                        ),
                    )
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

            with gr.Accordion("Extra CLI flags", open=False):
                reg(
                    "extra_flags",
                    gr.Textbox(
                        label="Raw extra args appended verbatim",
                        placeholder="--no-masked_loss --network_weights path.safetensors",
                    ),
                )

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

                # ── Run controls (kohya-style: Start at the very bottom) ──────
                # These live in the LoRA tab (not below the tabs) so the Utils
                # tab carries no "Start training" button.
                with gr.Row():
                    print_btn = gr.Button("Print training command", variant="secondary")
                    start_btn = gr.Button("Start training", variant="primary")
                    monitor_btn = gr.Button("Start monitoring", variant="secondary")
                    stop_btn = gr.Button("Stop", variant="stop")
                    status_btn = gr.Button("Refresh status", variant="secondary")

        with gr.Tab("Utils"):
            gr.Markdown(
                "Utilities run as **direct subprocesses** (`tasks.py …`), like "
                "training — **mutually exclusive** with a training run (one at a "
                "time). The launched command + result show in the **Result / status** "
                "box below; to cancel a running job use **Stop** on the LoRA tab."
            )

            # ── Update (GUI face of update.bat: git pull + uv sync) ─────────
            with gr.Accordion("Update (git pull + uv sync)", open=True):
                gr.Markdown(
                    "Update this tool to the latest commit on "
                    "**UR-al/training_Anima_lora** — the GUI equivalent of "
                    "`update.bat`. Your datasets / output / models are gitignored and "
                    "never touched. **Restart the GUI** after updating so the new code "
                    "loads. (Installed from a release zip instead of git? Use the "
                    "installer to update.)"
                )
                update_info = gr.Markdown("")
                with gr.Row():
                    check_update_btn = gr.Button(
                        "Check for updates", variant="secondary"
                    )
                    update_now_btn = gr.Button(
                        "Update now (git pull + uv sync)", variant="primary"
                    )
                update_log = gr.Textbox(
                    label="Update output",
                    lines=10,
                    interactive=False,
                    visible=False,
                )

            # ── Auto-batch search (tasks.py bench-autobatch) ────────────────
            with gr.Accordion("Auto-batch (max batch-size search)", open=True):
                _ab_tiers = [str(t) for t in server.list_target_res_tiers()]
                reg(
                    "ab_res",
                    gr.CheckboxGroup(
                        _ab_tiers,
                        value=[t for t in ("1024",) if t in _ab_tiers],
                        label="Resolutions to search (--res)",
                    ),
                )
                with gr.Row():
                    reg(
                        "ab_max_batch",
                        gr.Textbox(label="max batch (--max-batch)", placeholder="8"),
                    )
                    reg(
                        "ab_optimizer_type",
                        gr.Textbox(label="optimizer_type", placeholder="AdamW"),
                    )
                    reg(
                        "ab_blocks_to_swap",
                        gr.Textbox(label="blocks_to_swap", placeholder="0"),
                    )
                    reg("ab_compile", gr.Checkbox(value=False, label="--compile"))
                with gr.Row():
                    reg(
                        "ab_network_module",
                        gr.Textbox(
                            label="network_module", placeholder="networks.lora_anima"
                        ),
                    )
                    reg(
                        "ab_network_dim",
                        gr.Textbox(label="network_dim", placeholder="16"),
                    )
                    reg(
                        "ab_network_alpha",
                        gr.Textbox(label="network_alpha", placeholder="8"),
                    )
                reg_arg_rows("ab_network_args", title="Auto-batch network args")
                ab_run_btn = gr.Button("Run auto-batch", variant="primary")
            # ── Masking (tasks.py mask: SAM3 + MIT) ─────────────────────────
            with gr.Accordion("Masking (SAM3 + MIT)", open=False):
                gr.Markdown(
                    "Masks the configured resized dir → `post_image_dataset/masks/` "
                    "(dirs from `preprocess.toml`/`base.toml`). SAM3 needs "
                    "`models/sam3/`; MIT needs `models/mit/model.pth`."
                )
                with gr.Row():
                    reg(
                        "mask_sam", gr.Checkbox(value=True, label="SAM3 (RUN_SAM_MASK)")
                    )
                    reg("mask_mit", gr.Checkbox(value=True, label="MIT (RUN_MIT_MASK)"))
                    reg(
                        "mit_text_threshold",
                        gr.Textbox(label="MIT text threshold", placeholder="(default)"),
                    )
                    reg(
                        "mit_dilate",
                        gr.Textbox(label="MIT dilate", placeholder="(default)"),
                    )
                mask_run_btn = gr.Button("Run masking", variant="primary")

        # ── Shared command / result output (below both tabs) ────────────────
        # train.py's run controls (Print/Start/Stop/Status) live in the LoRA
        # tab's Monitor & Run accordion; these two outputs are shared so the
        # Utils run buttons (auto-batch / masking) can report into them too.
        # The live training log is the web monitor (--monitor) now — the old
        # terminal-tail panel was removed as redundant with it.
        out_cmd = gr.Code(label="train.py command", language="shell")
        out_status = gr.JSON(label="Result / status")

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

        def on_start_monitoring(*vals):
            """Open the web monitor: attach to a live --monitor run's URL, else spawn
            a standalone read-only dashboard (rehydrates the last run's curves)."""
            form = _collect(keys, vals)
            try:
                res = server.start_monitoring(form)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": str(exc)}
            url = res.get("url", "")
            return (f"# monitor: {url}" if url else ""), res

        def on_status():
            return server.status()

        def _fmt_version(v: dict) -> str:
            if not v.get("ok"):
                return f"ℹ️ {v.get('note', '')}"
            ahead = v.get("ahead", "0")
            ahead_note = (
                f" · {ahead} local commit(s) not pushed" if ahead != "0" else ""
            )
            head = (
                "✅ **Up to date.**"
                if v.get("up_to_date")
                else f"🔔 **{v['behind']} update(s) available** — click *Update now*."
            )
            return (
                f"{head}\n\n"
                f"- branch **{v['branch']}** @ `{v['sha']}`{ahead_note}\n"
                f"- this checkout: {v['last_commit']}\n"
                f"- latest on origin: {v.get('remote_last', '?')}\n"
                f"- remote: `{v.get('remote', '?')}`"
            )

        def on_check_update():
            return _fmt_version(server.tool_version(fetch=True))

        def on_update_now():
            res = server.update_tool()
            info = _fmt_version(server.tool_version(fetch=False))
            return info, gr.update(value=res.get("output", ""), visible=True)

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
            return [
                {"id": i.get("id"), "name": i.get("name")} for i in server.queue_list()
            ]

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
            for (
                group,
                n,
            ) in _ARG_ROW_GROUPS.items():  # inline key=value → name/value rows
                if group in form:
                    for i, (k, v) in enumerate(
                        _inline_to_arg_rows(form.pop(group), n), 1
                    ):
                        form[f"{group}__k{i}"] = k
                        form[f"{group}__v{i}"] = v
            # Fold the dep-greying interactive states into THIS load's output (no
            # secondary .change/.then event → no spinner-wedging race on load).
            interactive = _interactive_states(form)
            updates = []
            for k in keys:
                upd = {}
                if k in form:
                    upd["value"] = form[k]
                if k in interactive:
                    upd.update(interactive[k])
                updates.append(gr.update(**upd) if upd else gr.update())
            note = f"✓ loaded {len(form)} field(s) from `{p}`" + (
                " — unmapped keys are in *Extra CLI flags*"
                if form.get("extra_flags")
                else ""
            )
            overflow = form.get("_ds_overflow")
            if overflow:
                note += (
                    f" — ⚠ config has {overflow} subsets but the GUI shows "
                    f"{N_SUBSETS}; the extras were dropped. Use the `dataset_config` "
                    "field for that TOML to keep all of them."
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

        # ── Conflict / dependency greying (ported from the web GUI's CONFLICT_RULES
        #    + ARG_DEPS) — disable (and reset) a field when an active option makes it a
        #    no-op. Defined here so the config-load .then below can run ONE clean full
        #    recompute AFTER every value has landed. ──────────────────────────────────
        _dep_drivers = [
            "use_vae_cache",
            "use_text_cache",
            "use_constantcosine",
            "loss_type",
            "timestep_sampling",
            "weighting_scheme",
        ]
        _dep_targets = [
            "ds_random_crop",
            "ds_caption_dropout_rate",
            "lr_scheduler_type",
            "huber_c",
            "huber_schedule",
            "sigmoid_scale",
            "logit_mean",
            "logit_std",
        ]

        def _gray(on: bool, reset=None):
            if on:
                return gr.update(interactive=True)
            if reset is not None:
                return gr.update(interactive=False, value=reset)
            return gr.update(interactive=False)

        def _recompute_deps(use_vae, use_text, use_cc, loss, ts, ws):
            huber = loss in ("huber", "smooth_l1")
            return [
                _gray(not use_vae, reset=False),  # random_crop ↮ cached latents
                _gray(not use_text, reset="0"),  # caption_dropout ↮ cached TE
                _gray(not use_cc),  # lr_scheduler_type ← overridden by constant→cosine
                _gray(huber),  # huber_c only for huber/smooth_l1
                _gray(huber),  # huber_schedule
                _gray(ts in ("", "sigmoid")),  # sigmoid_scale: sigmoid family
                _gray(ws == "logit_normal"),  # logit_mean
                _gray(ws == "logit_normal"),  # logit_std
            ]

        _dep_in = [by_key[k] for k in _dep_drivers]
        _dep_out = [by_key[k] for k in _dep_targets]

        # Config load is a SINGLE event: on_load_config writes every value AND folds in
        # the dep-greying interactive states (via _interactive_states). NO .then / no
        # secondary recompute — the old .then raced the output-induced per-driver
        # .change cascade and left huber_c/sigmoid_scale/… stuck on a spinner. The
        # per-driver .change handlers (below) stay for live edits; on a load they just
        # recompute the identical state already applied (idempotent).
        load_cfg_btn.click(
            on_load_config, inputs=config_path, outputs=inputs + [config_status]
        )
        save_cfg_btn.click(
            on_save_config, inputs=[config_path, *inputs], outputs=config_status
        )

        print_btn.click(on_print, inputs=inputs, outputs=out_cmd)
        start_btn.click(on_start, inputs=inputs, outputs=[out_cmd, out_status])
        monitor_btn.click(
            on_start_monitoring, inputs=inputs, outputs=[out_cmd, out_status]
        )
        stop_btn.click(on_stop, inputs=None, outputs=[out_cmd, out_status])
        status_btn.click(on_status, inputs=None, outputs=out_status)
        ab_run_btn.click(on_autobatch, inputs=inputs, outputs=[out_cmd, out_status])
        mask_run_btn.click(on_masking, inputs=inputs, outputs=[out_cmd, out_status])
        check_update_btn.click(on_check_update, inputs=None, outputs=update_info)
        update_now_btn.click(
            on_update_now, inputs=None, outputs=[update_info, update_log]
        )
        queue_add_btn.click(on_queue_add, inputs=inputs, outputs=queue_view)
        queue_run_btn.click(on_queue_run, inputs=None, outputs=[queue_view, out_status])
        queue_refresh_btn.click(on_queue_refresh, inputs=None, outputs=queue_view)
        queue_clear_btn.click(on_queue_clear, inputs=None, outputs=queue_view)
        # `sample_prompts` is registered; locate its component to receive the path.
        sample_path_comp = inputs[keys.index("sample_prompts")]
        save_samples_btn.click(
            on_save_samples,
            inputs=[inputs[keys.index("output_name")], sample_editor],
            outputs=sample_path_comp,
        )

        # Per-driver SCOPED greying: each driver updates ONLY the target(s) it gates, so
        # toggling one (e.g. use_text_cache) no longer repaints/flickers the unrelated
        # huber/sigmoid/logit fields the old shared all-8-outputs handler rewrote. The
        # output-list length MUST match each lambda's return shape (1 update, or a list).
        bk = by_key
        bk["use_vae_cache"].change(
            lambda v: _gray(not v, reset=False),
            bk["use_vae_cache"],
            bk["ds_random_crop"],
        )
        bk["use_text_cache"].change(
            lambda v: _gray(not v, reset="0"),
            bk["use_text_cache"],
            bk["ds_caption_dropout_rate"],
        )
        bk["use_constantcosine"].change(
            lambda v: _gray(not v),
            bk["use_constantcosine"],
            bk["lr_scheduler_type"],
        )
        bk["loss_type"].change(
            lambda v: [
                _gray(v in ("huber", "smooth_l1")),
                _gray(v in ("huber", "smooth_l1")),
            ],
            bk["loss_type"],
            [bk["huber_c"], bk["huber_schedule"]],
        )
        bk["timestep_sampling"].change(
            lambda v: _gray(v in ("", "sigmoid")),
            bk["timestep_sampling"],
            bk["sigmoid_scale"],
        )
        bk["weighting_scheme"].change(
            lambda v: [_gray(v == "logit_normal"), _gray(v == "logit_normal")],
            bk["weighting_scheme"],
            [bk["logit_mean"], bk["logit_std"]],
        )
        # Initial interactive state: one full recompute on app load (the right place to
        # set all 8 at once); config-load uses the .then chain above for the same.
        demo.load(_recompute_deps, _dep_in, _dep_out)
        # Show the local version on startup WITHOUT a network fetch (offline-safe);
        # the "Check for updates" button does the fetch on demand.
        demo.load(
            lambda: _fmt_version(server.tool_version(fetch=False)), None, update_info
        )

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
