#!/usr/bin/env python
"""Multi-resolution training-step speed + VRAM bench for the Anima DiT.

Builds the *real* frozen DiT + a fresh plain LoRA, then runs a representative
training step (forward → flow-matching MSE → backward → optimizer step) on
SYNTHETIC tensors at each requested tier (512 / 1024 / 1536 …) and reports
**s/it, it/s and peak VRAM** per ``(tier, batch)``. No dataset / images / captions
needed — the synthetic latents + text-conditioning tensors use the exact training
shapes (latent ``(B,16,H/8,W/8)`` → 5D ``unsqueeze(2)`` into the DiT, crossattn
``(B,512,1024)``), so the grad-ckpt / block-swap / torch.compile deltas are
faithful. The DiT forward/backward dominates per-step cost; a plain LoRA stands in
for the adapter (its overhead is a small constant), so this measures the same
compute the real hot loop spends.

One MODEL config per invocation (grad-ckpt / block-swap / compile are set once at
build time); the run sweeps tiers × batch sizes inside that config. To compare
configs, run it a few times with different ``--label`` (the canonical bench idiom):

    # baseline (no grad-ckpt, no swap)
    python bench/speed/run_bench.py --tiers 512 1024 1536 --batch 1 2 --label base
    # gradient checkpointing on
    python bench/speed/run_bench.py --tiers 1536 --batch 1 2 --gradient_checkpointing --label gc
    # block swap on
    python bench/speed/run_bench.py --tiers 1024 --batch 2 --blocks_to_swap 10 --label swap10
    # with torch.compile (first step pays the compile cost; warmup covers it)
    python bench/speed/run_bench.py --tiers 1024 --batch 2 --compile --label compiled

Run on a FREE GPU — it loads the ~4 GB DiT. Each ``(tier, batch)`` that OOMs is
recorded as ``oom: true`` and the sweep continues. Results + a ``result.json``
envelope land under ``bench/speed/results/<ts>[-<label>]/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run as a script (python bench/speed/run_bench.py) puts bench/speed on sys.path[0],
# NOT the repo root — so root-level packages like LoraEasyCustomOptimizer (the
# optimizer zoo, not pip-installed) aren't importable and CAME/etc. would fall back
# to torch.optim. Put the repo root first so the zoo resolves like it does in train.py.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402 — after the repo-root sys.path bootstrap above
import torch.nn.functional as F  # noqa: E402

from bench._anima import add_common_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402


# --------------------------------------------------------------------------
# Tier → representative bucket → synthetic latent shape
# --------------------------------------------------------------------------


def _tier_bucket(edge: int) -> tuple[int, int, int, int, int]:
    """Return (W, H, W_lat, H_lat, token_count) for a tier's representative bucket.

    Uses the FIRST (nearest-to-square) bucket of each edge's table — the exact
    ``(W, H)`` the trainer would bucket an image of that tier into, so the
    synthetic latent matches a real cached one and the compile graph keys on the
    real token count. Latent = VAE/8; token count = (W//16)*(H//16) (patch=2).
    """
    from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS_BY_EDGE

    table = CONSTANT_TOKEN_BUCKETS_BY_EDGE.get(edge)
    if not table:
        allowed = sorted(CONSTANT_TOKEN_BUCKETS_BY_EDGE)
        raise SystemExit(f"--tiers {edge} not in bucket table; choose from {allowed}")
    w, h = table[0]
    return w, h, w // 8, h // 8, (w // 16) * (h // 16)


def _default_dit() -> str | None:
    """Resolve the base DiT path from the config chain so --dit is optional."""
    try:
        from library.config.io import load_method_preset

        cfg = load_method_preset("lora", "default")
        # base.toml uses `pretrained_model_name_or_path`; honor `dit` too.
        return cfg.get("pretrained_model_name_or_path") or cfg.get("dit")
    except Exception:  # noqa: BLE001 — best-effort default; --dit overrides
        return None


# --------------------------------------------------------------------------
# Model build (fresh plain LoRA on the frozen DiT, training-faithful ordering)
# --------------------------------------------------------------------------


def _build_model(args, device, dtype):
    """Frozen DiT + a fresh adapter (network_module + network_args), placed /
    checkpointed / compiled per args. So you can bench the REAL adapter — LoKr /
    LoHa via networks.lycoris_anima, the native LoRA family, … — not just plain LoRA.

    Mirrors the train.py / harness ordering: load → freeze → create_network →
    apply_to → place (block-swap) → grad-ckpt → train() → compile LAST.
    """
    import importlib

    from library.anima import weights as anima_utils
    from library.runtime.harness import (
        compile_dit_blocks_for_pool,
        place_dit_for_training,
    )

    # Block swap needs the DiT to start on CPU so placement can selectively move.
    loading_device = "cpu" if args.blocks_to_swap > 0 else device
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        loading_device=loading_device,
        dit_weight_dtype=dtype,
    )
    anima.requires_grad_(False)
    anima.reset_mod_guidance()

    # Fresh, untrained adapter from the chosen module + network_args (key=value
    # strings the module parses: algo, factor, full_matrix, preset, …). Mirrors
    # train.py's generic create_network call; empty args → plain LoRA.
    network_module = importlib.import_module(args.network_module)
    net_kwargs = {}
    for na in args.network_args or []:
        if "=" in na:
            k, v = na.split("=", 1)
            net_kwargs[k] = v
    network = network_module.create_network(
        1.0,
        args.network_dim,
        args.network_alpha,
        None,  # vae (unused)
        [None],  # text_encoders (unused — apply_text_encoder=False below)
        anima,
        neuron_dropout=0.0,
        **net_kwargs,
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=dtype)
    network.requires_grad_(True)

    # Place on device (+ arm forward/backward block swap when requested).
    place_dit_for_training(anima, device, blocks_to_swap=args.blocks_to_swap)

    # Configure the checkpointing machinery (offload variant) if grad-ckpt is on
    # globally OR per-resolution; the per-tier toggle in the measure loop then turns
    # the gate on/off so a tier checkpoints only when it should — exactly the
    # training-time per-resolution behavior.
    if args.gradient_checkpointing or args.gradient_checkpointing_resolutions:
        gc_kwargs = {}
        if args.unsloth_offload_checkpointing:
            gc_kwargs["unsloth_offload"] = True
        elif args.cpu_offload_checkpointing:
            gc_kwargs["cpu_offload"] = True
        anima.enable_gradient_checkpointing(**gc_kwargs)

    # train() BEFORE compile — Block.forward gates ckpt + the LoRA train path on
    # self.training, and compile must trace the train-mode monkey-patched forward.
    anima.train()
    network.train()

    # Compile LAST. Size the dynamo budget + seq-range to exactly the tiers this
    # run touches (the union of their token counts), so each shape traces once.
    token_counts = sorted({_tier_bucket(t)[4] for t in _active_tiers(args)})
    compile_dit_blocks_for_pool(
        anima,
        token_counts,
        enabled=args.compile,
        dynamic_seq=args.compile_dynamic_seq,
        mode=args.compile_mode,
        activation_memory_budget=args.activation_memory_budget,
        grad_ckpt=bool(args.gradient_checkpointing or args.gradient_checkpointing_resolutions),
    )
    return anima, network


def _tier_ckpt(args, tier: int) -> bool:
    """Whether THIS tier should gradient-checkpoint: the global flag checkpoints
    every tier; otherwise only the edges in --gradient_checkpointing_resolutions."""
    if args.gradient_checkpointing:
        return True
    res = args.gradient_checkpointing_resolutions
    return bool(res) and tier in res


def _active_tiers(args) -> list:
    """The tier edges this run touches (sizes the compile token budget)."""
    if args.plan:
        return [int(p.split(":")[0]) for p in args.plan]
    return list(args.tiers)


def _plan_cells(args) -> list:
    """The (tier, batch) cells to measure. --plan gives explicit per-tier batches
    (e.g. 512:4 1024:2 1536:1); otherwise the tiers x batch cartesian."""
    if args.plan:
        return [(int(p.split(":")[0]), int(p.split(":")[1])) for p in args.plan]
    return [(t, b) for t in args.tiers for b in args.batch]


def _measure_cell(anima, network, optimizer, args, tier, batch, device, dtype, cuda, gc_on):
    """Warmup + timed run for one (tier, batch); returns its result record (or an
    oom record). Catches OOM so a sweep continues; always frees afterward."""
    w, h, w_lat, h_lat, tokens = _tier_bucket(tier)
    tag = f"{tier}@b{batch}"
    try:
        bat = _make_batch(batch, w_lat, h_lat, device, dtype)
        for _ in range(args.warmup):  # warmup (compile + caches), untimed
            _step(anima, network, optimizer, bat, device, dtype, args.blocks_to_swap)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.steps):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _step(anima, network, optimizer, bat, device, dtype, args.blocks_to_swap)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times.sort()
        median = times[len(times) // 2]
        mean = sum(times) / len(times)
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**2 if cuda else 0.0
        peak_resv = torch.cuda.max_memory_reserved() / 1024**2 if cuda else 0.0
        print(f"  {tag:>12}  {tokens:>5} tok{' +gc' if gc_on else '    '}  {median:6.3f} s/it  "
              f"{1.0 / median:6.2f} it/s  peak {peak_resv / 1024:5.2f} GiB")
        del bat
        return {
            "tier": tier, "w": w, "h": h, "tokens": tokens,
            "latent_shape": [batch, 16, h_lat, w_lat], "batch": batch,
            "mean_s_per_it": mean, "median_s_per_it": median,
            "std_s_per_it": (sum((x - mean) ** 2 for x in times) / len(times)) ** 0.5,
            "it_per_s": 1.0 / median if median > 0 else None,
            "peak_alloc_mib": round(peak_alloc, 1),
            "peak_reserved_mib": round(peak_resv, 1),
            "grad_ckpt": gc_on, "oom": False,
        }
    except torch.cuda.OutOfMemoryError:
        print(f"  {tag:>12}  {tokens:>5} tok{' +gc' if gc_on else '    '}  OOM")
        return {"tier": tier, "batch": batch, "tokens": tokens, "grad_ckpt": gc_on, "oom": True}
    finally:
        optimizer.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------
# One synthetic training step (matches train.py's cached-TE forward call)
# --------------------------------------------------------------------------


def _make_batch(batch, w_lat, h_lat, device, dtype):
    """Synthetic cached-path batch: 5D noisy latent, timesteps, crossattn, mask."""
    latents = torch.randn(batch, 16, h_lat, w_lat, device=device, dtype=dtype)
    noise = torch.randn_like(latents)
    sigma = torch.rand(batch, 1, 1, 1, device=device, dtype=dtype)  # flow time σ∈[0,1]
    noisy = (1.0 - sigma) * latents + sigma * noise
    return {
        "noisy_5d": noisy.unsqueeze(2),  # (B,16,1,H,W) — the DiT's dim-2 singleton
        "timesteps": sigma.view(batch, 1).float(),  # (B,1), ndim==2 invariant
        "crossattn_emb": torch.randn(batch, 512, 1024, device=device, dtype=dtype),
        "padding_mask": torch.zeros(batch, 1, h_lat, w_lat, device=device, dtype=dtype),
        "target": noise - latents,  # (B,16,H,W) rectified-flow target (4D)
    }


def _build_optimizer(args, network):
    """Real optimizer factory so the choice (AdamW / 8-bit / CAME …) shows up in
    VRAM faithfully; falls back to torch AdamW if the factory needs a field we
    don't populate. Optimizer state is second-order for LoRA, so the fallback is
    a safe representative."""
    import argparse as _ap
    import ast as _ast

    params = [p_ for p_ in network.parameters() if p_.requires_grad]
    otype = (args.optimizer_type or "AdamW").strip()
    okw = {}
    for a in args.optimizer_args or []:
        if "=" in a:
            k, v = a.split("=", 1)
            try:
                okw[k] = _ast.literal_eval(v)
            except (ValueError, SyntaxError):
                okw[k] = v
    if otype.lower() == "adamw" and not okw:
        return torch.optim.AdamW(params, lr=args.learning_rate)

    # 1) the real kohya factory (friendly-name resolution through the optimizer zoo).
    try:
        from library.training.optimizers import get_optimizer

        ns = _ap.Namespace(
            optimizer_type=otype,
            optimizer_args=list(args.optimizer_args or []),
            learning_rate=args.learning_rate,
            lr=args.learning_rate,
            fused_backward_pass=False,
            gradient_accumulation_steps=1,
            max_grad_norm=0.0,
        )
        return get_optimizer(ns, params)[2]
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimizer] get_optimizer({otype}) failed ({exc})", flush=True)

    # 2) resolve the class straight from the zoo — covers the case where the factory
    #    fell through to torch.optim because the zoo registry was momentarily empty.
    try:
        from LoraEasyCustomOptimizer import OPTIMIZERS

        cls = OPTIMIZERS.get(otype.lower())
        if cls is not None:
            print(f"  [optimizer] resolved {otype} directly from the optimizer zoo", flush=True)
            return cls(params, lr=args.learning_rate, **okw)
        print(f"  [optimizer] '{otype}' not in the zoo ({len(OPTIMIZERS)} available)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [optimizer] optimizer-zoo import failed: {exc}", flush=True)

    # 3) AdamW fallback — for LoRA the optimizer state is tiny, so the VRAM frontier
    #    is ~unchanged vs CAME/etc.; only the per-step optimizer cost differs slightly.
    print(f"  [optimizer] using AdamW instead of {otype} (VRAM ~ same for LoRA)", flush=True)
    return torch.optim.AdamW(params, lr=args.learning_rate)


def _step(anima, network, optimizer, batch, device, dtype, blocks_to_swap):
    """forward → squeeze → MSE(flow target) → backward → opt.step. Matches the
    real cached-TE call: anima(noisy_5d, timesteps, crossattn_emb, padding_mask=…)."""
    if blocks_to_swap > 0:
        anima.prepare_block_swap_before_forward(free_cache=False)
    with torch.autocast(device_type=device.type, dtype=dtype):
        pred = anima(
            batch["noisy_5d"],
            batch["timesteps"],
            batch["crossattn_emb"],
            padding_mask=batch["padding_mask"],
        )
    pred = pred.squeeze(2)  # 5D → 4D
    loss = F.mse_loss(pred.float(), batch["target"].float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # --label/--seed/--device/--dtype/--attn_mode/--gradient_checkpointing/
    # --cpu_offload_checkpointing/--compile/--compile_mode:
    add_common_args(p)
    p.add_argument("--dit", default=_default_dit(), help="Base DiT safetensors (default: config chain).")
    p.add_argument("--tiers", type=int, nargs="+", default=[512, 1024, 1536],
                   help="Resolution edges to sweep (512 768 896 1024 1280 1536).")
    p.add_argument("--batch", type=int, nargs="+", default=[1],
                   help="Batch size(s) swept across all --tiers (cartesian).")
    p.add_argument("--plan", type=str, nargs="*", default=None,
                   help="Per-tier batches as tier:batch pairs (e.g. 512:4 1024:2 1536:1), "
                   "overriding the --tiers x --batch cartesian — matches a real multiscale run.")
    p.add_argument("--steps", type=int, default=8, help="Timed steps per (tier,batch).")
    p.add_argument("--warmup", type=int, default=3, help="Untimed warmup steps (covers compile).")
    p.add_argument("--blocks_to_swap", type=int, default=0, help="DiT blocks to CPU-swap (0=off, max num_blocks-2).")
    p.add_argument("--gradient_checkpointing_resolutions", type=int, nargs="*", default=None,
                   help="Per-resolution gradient checkpointing (matches training): checkpoint ONLY "
                   "these tier edges (e.g. 1536), so the big tier fits while the smaller tiers stay "
                   "full-speed. The per-tier gate is toggled in the measure loop. --gradient_checkpointing "
                   "(global) still checkpoints every tier.")
    p.add_argument("--unsloth_offload_checkpointing", action="store_true",
                   help="With gradient checkpointing, use the Unsloth async CPU-offload variant.")
    p.add_argument("--activation_memory_budget", type=float, default=1.0,
                   help="torch.compile partitioner saved-activation fraction (<1.0; ignored under grad-ckpt).")
    p.add_argument("--compile_dynamic_seq", action="store_true",
                   help="Compile one symbolic-seq graph instead of one per token count.")
    p.add_argument("--network_dim", type=int, default=16,
                   help="Adapter rank. LoKr idiom: 100000 (+ factor in --network_args).")
    p.add_argument("--network_alpha", type=float, default=8.0, help="Adapter alpha.")
    p.add_argument("--network_module", type=str, default="networks.lora_anima",
                   help="Adapter module — e.g. networks.lycoris_anima for LoKr / LoHa.")
    p.add_argument("--network_args", type=str, nargs="*", default=[],
                   help="key=value adapter args, e.g. algo=lokr factor=4 full_matrix=True preset=<toml>.")
    p.add_argument("--optimizer_type", type=str, default="AdamW",
                   help="Optimizer (real factory; e.g. AdamW, AdamW8bit, CAME). Affects VRAM faithfully.")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer_args", type=str, nargs="*", default=[],
                   help="key=value optimizer args forwarded to the factory.")
    p.add_argument("--out_json", type=str, default=None,
                   help="Also write the metrics record to this exact path (for the sweep orchestrator).")
    args = p.parse_args()

    if not args.dit:
        raise SystemExit("no DiT path: pass --dit <anima_dit.safetensors> (config default unavailable).")

    device = torch.device(args.device)
    from library.runtime.device import str_to_dtype

    dtype = str_to_dtype(args.dtype)
    torch.manual_seed(args.seed)
    cuda = device.type == "cuda"

    run_dir = make_run_dir("speed", label=args.label)
    gc_desc = (args.gradient_checkpointing_resolutions or args.gradient_checkpointing)
    print(f"building model (dit={args.dit}, dim={args.network_dim}, swap={args.blocks_to_swap}, "
          f"grad_ckpt={gc_desc}, compile={args.compile})…")
    anima, network = _build_model(args, device, dtype)
    optimizer = _build_optimizer(args, network)
    n_train = sum(p_.numel() for p_ in network.parameters() if p_.requires_grad)
    print(f"  trainable adapter params: {n_train:,} ({args.network_module})")

    runs = []
    for tier, batch in _plan_cells(args):
        gc_on = _tier_ckpt(args, tier)
        anima.set_gradient_checkpointing(gc_on)  # per-resolution ckpt gate
        runs.append(_measure_cell(
            anima, network, optimizer, args, tier, batch, device, dtype, cuda, gc_on
        ))

    metrics = {
        "config": {
            "dtype": args.dtype, "attn_mode": args.attn_mode,
            "gradient_checkpointing": args.gradient_checkpointing,
            "gradient_checkpointing_resolutions": args.gradient_checkpointing_resolutions,
            "cpu_offload_checkpointing": args.cpu_offload_checkpointing,
            "unsloth_offload_checkpointing": args.unsloth_offload_checkpointing,
            "blocks_to_swap": args.blocks_to_swap,
            "compile": args.compile, "compile_mode": args.compile_mode,
            "compile_dynamic_seq": args.compile_dynamic_seq,
            "activation_memory_budget": args.activation_memory_budget,
            "network_dim": args.network_dim, "trainable_params": n_train,
            "steps": args.steps, "warmup": args.warmup,
        },
        "runs": runs,
    }
    out = write_result(run_dir, script=__file__, args=args, metrics=metrics,
                       label=args.label, device=device)
    if args.out_json:
        import json as _json
        import os as _os

        _os.makedirs(_os.path.dirname(_os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as _f:
            _json.dump(metrics, _f, indent=2)
    print(f"\nresult → {out}")


if __name__ == "__main__":
    main()
