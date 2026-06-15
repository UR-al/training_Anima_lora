# -*- coding: utf-8 -*-
"""Gradio Blocks UI for the Anima LoRA trainer.

Layout modelled on the kohya_ss GUI (source-model paths + an Anima accordion +
basic / network / optimizer / dataset / sample / advanced sections), but every
control feeds the ``form`` dict consumed by :mod:`scripts.webgui.server`, whose
``build_command`` emits the exact ``train.py --method … --preset …`` invocation
and whose ``launch`` submits it to the training daemon. No kohya ``sd-scripts``
code path is involved — the GUI is a front-end re-skin over our own backend.
"""

from __future__ import annotations

from scripts.webgui import server

# --------------------------------------------------------------------------- #
# Form-field registry: the click handlers receive positional values, so the
# order of FIELD_KEYS MUST match the order components are appended to `inputs`.
# Each value lands verbatim in the `form` dict server._method_preset_extra /
# server.launch already understand — blanks fall back to the config chain.
# --------------------------------------------------------------------------- #
FIELD_KEYS: list[str] = []


def _register(keys: list[str], comps: list, key: str, comp):
    """Append one component, recording its form key in lockstep."""
    keys.append(key)
    comps.append(comp)
    return comp


def _collect(keys: list[str], values) -> dict:
    """Zip positional handler args back into the backend `form` dict."""
    form = dict(zip(keys, values))
    # Gradio Textbox yields "" for empty; backend treats "" as "use default".
    # Coerce checkbox-style truthiness through untouched (already bool).
    return form


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

    def reg(key, comp):
        return _register(keys, inputs, key, comp)

    with gr.Blocks(title="Anima LoRA Trainer") as demo:
        gr.Markdown(
            "# Anima LoRA Trainer\n"
            "Gradio front-end (kohya-style layout) driving **this repo's** "
            "`train.py --method <name> --preset <name>`. Blank fields defer to the "
            "`base.toml → preset → method` config chain. Start submits to the "
            "training daemon (survives closing this tab)."
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
                    reg(
                        "output_dir",
                        gr.Textbox(
                            value="output",
                            label="Output base dir",
                        ),
                    )

            # ── Anima model paths (mirrors kohya's class_anima accordion) ────
            with gr.Accordion(
                "Anima Model Paths (blank = config-chain default)", open=False
            ):
                reg(
                    "dit_path",
                    gr.Textbox(
                        label="DiT checkpoint (--pretrained_model_name_or_path)",
                        placeholder="Path to the Anima DiT .safetensors",
                    ),
                )
                reg(
                    "te_path",
                    gr.Textbox(
                        label="Qwen3 text encoder (--qwen3)",
                        placeholder="Path to Qwen3-0.6B model dir / .safetensors",
                    ),
                )
                reg(
                    "vae_path",
                    gr.Textbox(
                        label="VAE (--vae)",
                        placeholder="Path to the Qwen-Image VAE",
                    ),
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
            with gr.Accordion("Dataset", open=False):
                reg(
                    "dataset_config",
                    gr.Textbox(
                        label="Dataset config TOML (blank = base.toml blueprint)",
                        placeholder="path/to/dataset.toml",
                    ),
                )
                gr.Markdown(
                    "_Leave blank to use the default `base.toml` dataset blueprint "
                    "(`post_image_dataset/lora` caches). Point at a `--dataset_config` "
                    "TOML for a custom blueprint. Run `make preprocess` first._"
                )

            # ── Sample prompts ──────────────────────────────────────────────
            with gr.Accordion("Sample images", open=False):
                reg(
                    "sample_prompts",
                    gr.Textbox(
                        label="Sample prompts file (--sample_prompts)",
                        placeholder="path/to/prompts.txt",
                    ),
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
                        "daemon",
                        gr.Checkbox(
                            value=True,
                            label="Run via daemon (detached; survives close)",
                        ),
                    )
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

            with gr.Accordion("Extra CLI flags", open=False):
                reg(
                    "extra_flags",
                    gr.Textbox(
                        label="Raw extra args appended verbatim",
                        placeholder="--max_grad_norm 1.0 --gradient_checkpointing",
                    ),
                )

            # ── Actions ─────────────────────────────────────────────────────
            with gr.Row():
                print_btn = gr.Button("Print training command", variant="secondary")
                start_btn = gr.Button("Start training", variant="primary")
                stop_btn = gr.Button("Stop", variant="stop")
                status_btn = gr.Button("Refresh status", variant="secondary")

            out_cmd = gr.Code(label="train.py command", language="shell")
            out_status = gr.JSON(label="Result / status")

            # ── Live training log (the daemon-captured terminal output) ─────
            # train.py's console log is the sd-scripts RichHandler format; when
            # the run goes through the daemon, its stdout/stderr is captured to
            # <job>/stdout.log, which server.log_tail() exposes. This panel
            # mirrors the terminal live so the GUI and the console stay linked.
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

        print_btn.click(on_print, inputs=inputs, outputs=out_cmd)
        start_btn.click(on_start, inputs=inputs, outputs=[out_cmd, out_status])
        stop_btn.click(on_stop, inputs=None, outputs=[out_cmd, out_status])
        status_btn.click(on_status, inputs=None, outputs=out_status)
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

    # Stash the field order on the app for any callers / debugging.
    FIELD_KEYS[:] = keys
    return demo


def serve(host: str = "127.0.0.1", port: int = 7860, open_browser: bool = True) -> None:
    """Launch the Gradio server (blocking)."""
    demo = build_app(default_port=port)
    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        show_api=False,
    )
