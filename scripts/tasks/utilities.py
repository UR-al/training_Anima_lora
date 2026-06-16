"""Misc utility entry-points: merge, comfy-batch, distill-prep, distill-mod,
test-unit, update, export-logs, print-config, bench-speed, bench-sweep,
bench-autobatch."""

from __future__ import annotations

import os

from ._common import PY, _preset, bespoke_preset_flags, run


def cmd_merge(extra):
    """Bake latest LoRA in ADAPTER_DIR (env, default 'output/ckpt') into the base DiT."""
    adapter_dir = os.environ.get("ADAPTER_DIR", "output/ckpt")
    multiplier = os.environ.get("MULTIPLIER", "1.0")
    run(
        [
            PY,
            "tools/merge_to_dit.py",
            "--adapter_dir",
            adapter_dir,
            "--multiplier",
            multiplier,
            *extra,
        ]
    )


def cmd_comfy_batch(extra):
    """Run a ComfyUI workflow as a batch.

    Workflow via ``W=`` (bare names resolve under ``workflows/``) or positional
    ``ARGS``. For workflows with a ``LoadImage`` node, ``IMAGES=<dir>`` switches
    on per-image sequential mode (default ``../comfy/input/to_colorize``):

        make comfy-batch W=colorize.json
        make comfy-batch W=colorize.json IMAGES=/path/to/imgs
    """
    workflow = os.environ.get("W") or (
        extra[0] if extra else "workflows/modhydra-simple.json"
    )
    if os.sep not in workflow and "/" not in workflow:
        workflow = f"workflows/{workflow}"
    remaining = extra[1:] if (extra and not os.environ.get("W")) else list(extra)

    images_dir = os.environ.get("IMAGES", "../comfy/input/to_colorize")
    if images_dir and "--images_dir" not in remaining:
        remaining = ["--images_dir", images_dir, *remaining]

    run([PY, "scripts/comfy_batch.py", workflow, *remaining])


def cmd_distill_prep(extra):
    """Pre-stage artifacts for ``make distill-mod``.

    Phase 1: emits ``post_image_dataset/_anima_uncond_te.safetensors``
    (T5("") cross-attn baseline) — consumed as the student's unconditional
    text input, replacing the zeroed-crossattn shortcut. ``make preprocess-te``
    already produces this for free; this Phase 1 block is the explicit
    re-stager (useful with ``--overwrite`` after a model swap).

    Phase 2: emits teacher-synthesized clean latents under
    ``post_image_dataset/distill_mod_synth/`` (same NPZ layout as
    ``cache_latents.py``). Train with
    ``make distill-mod ARGS='--synth_data_dir post_image_dataset/distill_mod_synth'``
    to fit on the teacher's manifold (paper-faithful; removes real-vs-teacher
    gap that floors val loss).

    Skip flags forwarded via ``extra``: ``--skip_uncond``, ``--skip_synth``,
    ``--max_samples N``, etc.
    """
    run([PY, "-m", "scripts.distill_mod.prep", *extra])


def cmd_distill_mod(extra):
    """Distill the pooled_text_proj MLP for modulation guidance.

    Honors ``PRESET`` (default ``default``) — translates ``blocks_to_swap`` and
    ``gradient_checkpointing`` from ``configs/presets.toml`` into CLI flags so
    ``make distill-mod PRESET=low_vram`` enables grad ckpt + unsloth offload.
    Trailing ``extra`` args are appended last, so user CLI overrides win.

    Saves to ``output/ckpt/pooled_text_proj.safetensors`` so ``make test MOD=1``
    picks it up automatically.
    """
    preset_flags = bespoke_preset_flags(_preset())
    run(
        [
            PY,
            "-m",
            "scripts.distill_mod.distill",
            "--data_dir",
            "post_image_dataset/lora",
            "--dit_path",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--output_path",
            "output/ckpt/pooled_text_proj.safetensors",
            "--attn_mode",
            "flash",
            *preset_flags,
            *extra,
        ]
    )


def cmd_vendor_sync(extra):
    """Refresh custom_nodes/*/_vendor/ trees from the live library.* sources.

    Run before bumping a custom-node version / publishing — the bundled
    vendor copies (tagger + directedit) are how the ComfyUI nodes import
    their inference subset when not running inside the anima_lora repo.
    """
    run([PY, "scripts/sync_vendor.py", *extra])


def cmd_monitor(extra):
    """Standalone web monitor — browse saved run snapshots / replay a finished run
    offline and compare losses (no training). python tasks.py monitor
    [--output_dir output] [--port 8766] [--no-browser]."""
    run([PY, "tools/run_monitor.py", *(extra or [])])


def cmd_export_logs(extra):
    """Dump TB scalar logs to JSON. RUN=<dir> (default output/logs), ALL=1, JSONL=1."""
    run_path = os.environ.get("RUN", "output/logs")
    cmd = [PY, "scripts/export_logs_json.py", run_path]
    if os.environ.get("ALL"):
        cmd.append("--all")
    if os.environ.get("JSONL"):
        cmd.append("--jsonl")
    run([*cmd, *extra])


def cmd_print_config(extra):
    method = os.environ.get("METHOD", "lora")
    preset = _preset()
    run(
        [
            PY,
            "train.py",
            "--method",
            method,
            "--preset",
            preset,
            "--print-config",
            "--no-config-snapshot",
            *extra,
        ]
    )


def cmd_bench_speed(extra):
    """Multi-resolution training-step speed + VRAM bench (synthetic, no dataset).

    Sweeps tiers x batch inside one model config; run a few times with different
    --label to compare grad-ckpt / block-swap / compile. Needs a free GPU.
    e.g. python tasks.py bench-speed --tiers 512 1024 1536 --batch 1 2 --label base
    """
    run([PY, "bench/speed/run_bench.py", *extra])


def cmd_bench_sweep(extra):
    """XYZ-grid OOM + speed explorer: sweep budget / blocks_to_swap / batch /
    grad-ckpt / compile to find the max feasible config per resolution. Each cell
    runs isolated (fresh CUDA allocator → clean OOM frontier). Needs a free GPU.
    e.g. python tasks.py bench-sweep --res 1024 1536 --batch 1 2 --grad-ckpt on off
    --blocks-to-swap range:0-26:4 [--dry-run]. EXTRA after -- forwards to each cell.
    """
    run([PY, "bench/speed/sweep.py", *extra])


def cmd_bench_autobatch(extra):
    """Auto-find the MAX feasible batch per resolution (binary search, OOM-isolated).
    Give --res / --gradient_checkpointing_resolutions / --network_module / --network_args
    / --optimizer_type and it reports the largest batch each resolution holds without
    OOM + the s/it. Each trial is its own subprocess (clean OOM boundary). Free GPU.
    e.g. python tasks.py bench-autobatch --res 512 1024 1536 --max-batch 8
    --gradient_checkpointing_resolutions 1536 --network_module networks.lycoris_anima
    --network_dim 100000 --network_alpha 1 --network_args algo=lokr factor=4
    full_matrix=True preset=configs/lycoris_presets/anima_attn_mlp.toml --optimizer_type CAME
    """
    run([PY, "bench/speed/autobatch.py", *extra])
