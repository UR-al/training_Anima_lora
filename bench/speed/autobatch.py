#!/usr/bin/env python
"""Auto-find the MAX feasible batch per resolution (binary search, OOM-isolated).

You give the resolutions + which ones use gradient checkpointing + the network
(LoKr/LoHa/LoRA) + optimizer; this binary-searches the largest batch each
resolution can hold WITHOUT OOM and reports it with the s/it and peak VRAM. Each
trial runs run_bench.py in its OWN subprocess so a fresh CUDA allocator gives a
clean OOM boundary (fragmentation across trials can't move the line). The adapter /
optimizer / per-resolution checkpointing all match a real multiscale run, and no
images are needed (synthetic latents — VRAM/speed are shape-determined).

The search is monotonic (if batch N fits, N-1 fits), so binary search over
[1, --max-batch] is exact: ~log2(max_batch)+1 trials per resolution.

Example — your LoKr + CAME multiscale plan:
  python tasks.py bench-autobatch --res 512 1024 1536 --max-batch 8 \
    --gradient_checkpointing_resolutions 1536 \
    --network_module networks.lycoris_anima --network_dim 100000 --network_alpha 1 \
    --network_args algo=lokr factor=4 full_matrix=True \
    preset=configs/lycoris_presets/anima_attn_mlp.toml \
    --optimizer_type CAME

Needs a FREE GPU (each trial loads the DiT). Anything after -- is forwarded to
every run_bench trial. Result → bench/speed/results/<ts>[-<label>]/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bench._common import REPO_ROOT, make_run_dir, write_result

RUN_BENCH = Path(__file__).resolve().parent / "run_bench.py"
_TRIAL_LOG: list = []  # every trial's one-line result, replayed as a summary at the end


def _trial(args, res, batch, swap, out_json, extra, budget=None):
    """Run one (res, batch, swap, budget) trial via run_bench; return (fits, rec).

    ``budget`` overrides args.activation_memory_budget (for the budget auto-search).
    rec carries median_s_per_it / peak_reserved_mib / grad_ckpt when it fit. A
    missing out_json (fatal/uncatchable OOM crash) counts as not-fit.
    """
    budget = args.activation_memory_budget if budget is None else budget
    cmd = [
        sys.executable, str(RUN_BENCH),
        "--device", args.device, "--dtype", args.dtype, "--attn_mode", args.attn_mode,
        "--seed", str(args.seed),
        "--plan", f"{res}:{batch}",
        "--steps", str(args.steps), "--warmup", str(args.warmup),
        "--blocks_to_swap", str(swap),
        "--activation_memory_budget", str(budget),
        "--network_module", args.network_module,
        "--network_dim", str(args.network_dim), "--network_alpha", str(args.network_alpha),
        "--optimizer_type", args.optimizer_type, "--learning_rate", str(args.learning_rate),
        "--out_json", str(out_json), "--label", f"ab_{res}_{batch}",
    ]
    if args.dit:
        cmd += ["--dit", args.dit]
    if args.network_args:
        cmd += ["--network_args", *args.network_args]
    if res in (args.gradient_checkpointing_resolutions or []):
        cmd += ["--gradient_checkpointing_resolutions", str(res)]
    if args.compile:
        cmd.append("--compile")
    cmd += extra

    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    rec = None
    if out_json.exists():
        try:
            runs = json.loads(out_json.read_text()).get("runs", [])
            rec = runs[0] if runs else None
        except Exception:  # noqa: BLE001
            rec = None
    if rec is None:  # crash / fatal OOM before a record was written
        rec = {"tier": res, "batch": batch, "oom": True, "crashed": proc.returncode != 0}
    fits = not rec.get("oom")

    # Human-readable per-trial line: the exact config + OOM-or-success-with-speed.
    gc = res in (args.gradient_checkpointing_resolutions or [])
    net = args.network_module.split(".")[-1]
    bud = f" bud={budget}" if budget < 1.0 else ""
    cmp = " +compile" if args.compile else ""
    cfg = (f"res={res} batch={batch} swap={swap}" + (" +gc" if gc else "") + bud + cmp
           + f"  [{net} / {args.optimizer_type}]")
    gib = (rec.get("peak_reserved_mib") or 0) / 1024
    if fits:
        sit = rec.get("median_s_per_it") or 0.0
        # throughput (img/s = batch / s_it) is THE metric for "fastest multiscale".
        speed = f"{sit:.3f} s/it, {batch / sit:.2f} img/s" if sit else "측정됨"
        line = f"  [성공] {cfg}  ->  {speed}, peak {gib:.1f} GiB"
    else:
        # peak isn't captured on a caught OOM (it dies mid-step) → don't show 0.0.
        line = f"  [OOM ] {cfg}  ->  이 설정으로 OOM 났습니다"
    print(line, flush=True)
    _TRIAL_LOG.append(line)
    return fits, rec


def _max_batch(args, res, swap, cell_dir, extra, budget=None):
    """Binary-search the largest batch in [1, --max-batch] that fits at ``res`` with
    ``swap`` blocks swapped and ``budget`` activation budget. Returns (batch, rec) of
    the best fit, or None if even batch 1 OOMs."""
    lo, hi, best = 1, args.max_batch, None
    while lo <= hi:
        mid = (lo + hi) // 2
        tag = cell_dir / f"r{res}_s{swap}_bud{budget}_b{mid}.json"
        fits, rec = _trial(args, res, mid, swap, tag, extra, budget=budget)
        if fits:
            best = (mid, rec)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _budget_grid(args) -> list:
    """Activation-budget candidates: --min-budget up to 0.95 in 0.05 steps, + 0.99
    (ascending). Fit is monotonic — lower budget recomputes more → frees more VRAM."""
    g, b = [], args.min_budget
    while b <= 0.951:
        g.append(round(b, 2))
        b += 0.05
    if 0.99 not in g:
        g.append(0.99)
    return g


def _search_budget(args, res, cell_dir, extra):
    """Find (batch, swap, budget, rec) using activation_memory_budget as the fit lever
    (needs compile; main() forces it on). Phase 1: max batch at the LOWEST budget
    (most VRAM relief). Phase 2: for that batch, the HIGHEST budget (= least slowdown)
    that still fits. Returns None if infeasible even at the lowest budget."""
    swap = args.blocks_to_swap
    grid = _budget_grid(args)
    best = _max_batch(args, res, swap, cell_dir, extra, budget=grid[0])  # lowest budget
    if best is None:
        return None
    batch = best[0]
    # highest budget that still fits this batch (binary over the ascending grid).
    lo, hi, hi_bud, hi_rec = 0, len(grid) - 1, grid[0], best[1]
    while lo <= hi:
        mid = (lo + hi) // 2
        bud = grid[mid]
        tag = cell_dir / f"r{res}_b{batch}_bud{bud}.json"
        fits, rec = _trial(args, res, batch, swap, tag, extra, budget=bud)
        if fits:
            hi_bud, hi_rec = bud, rec
            lo = mid + 1
        else:
            hi = mid - 1
    return batch, swap, hi_bud, hi_rec


def _search(args, res, cell_dir, extra):
    """Find (batch, swap, rec) for this resolution. Max batch at the BASE swap
    (--blocks_to_swap); if even batch 1 OOMs there and --max-swap permits it,
    binary-search the MINIMAL blocks_to_swap that fits batch 1 (swap is slow → use
    the least needed; fit is monotonic in swap), then the max batch at that swap.
    Returns None if infeasible even at --max-swap."""
    base = args.blocks_to_swap
    bud = args.activation_memory_budget
    best = _max_batch(args, res, base, cell_dir, extra)
    if best is not None:
        return best[0], base, bud, best[1]
    if args.max_swap <= base:
        return None
    lo, hi, fit_swap, fit_rec = base + 1, args.max_swap, None, None
    while lo <= hi:
        mid = (lo + hi) // 2
        fits, rec = _trial(args, res, 1, mid, cell_dir / f"r{res}_s{mid}_b1.json", extra)
        if fits:
            fit_swap, fit_rec = mid, rec
            hi = mid - 1
        else:
            lo = mid + 1
    if fit_swap is None:
        return None
    # b1 fits at fit_swap, so this never returns None.
    best = _max_batch(args, res, fit_swap, cell_dir, extra)
    return (best[0], fit_swap, bud, best[1]) if best else (1, fit_swap, bud, fit_rec)


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--res", type=int, nargs="+", default=[512, 1024, 1536],
                   help="Resolution edges to search.")
    p.add_argument("--max-batch", dest="max_batch", type=int, default=8,
                   help="Upper bound of the per-resolution batch search.")
    p.add_argument("--gradient_checkpointing_resolutions", type=int, nargs="*", default=None,
                   help="Resolutions that use gradient checkpointing during the search.")
    p.add_argument("--blocks_to_swap", type=int, default=0,
                   help="Base blocks_to_swap for every trial (0 = none).")
    p.add_argument("--max-swap", dest="max_swap", type=int, default=0,
                   help="If >0, when a resolution OOMs at --blocks_to_swap, AUTO-escalate "
                   "blocks_to_swap up to this (<=26) to find the minimal swap that fits batch "
                   "1, then the max batch at that swap. 0 = don't auto-search swap.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the blocks (needed for --activation_memory_budget; "
                   "matches real training).")
    p.add_argument("--activation_memory_budget", type=float, default=1.0,
                   help="torch.compile partitioner activation fraction (<1.0 recomputes cheap "
                   "intermediates → less VRAM, mild slowdown). Needs --compile, ignored under "
                   "grad-ckpt. THIS is the lever base anima_lora uses to fit big batches eager "
                   "OOMs — pass e.g. 0.4 + --compile to match real training.")
    p.add_argument("--auto-budget", dest="auto_budget", action="store_true",
                   help="AUTO-search activation_memory_budget: per resolution find the max batch "
                   "at the lowest budget, then the HIGHEST budget (= least slowdown) that still "
                   "fits it. Implies --compile. Reports (batch, budget) per resolution.")
    p.add_argument("--min-budget", dest="min_budget", type=float, default=0.1,
                   help="Lowest budget the --auto-budget search tries (most VRAM relief). 0.05 step.")
    # adapter + optimizer (forwarded to each run_bench trial)
    p.add_argument("--network_module", default="networks.lora_anima")
    p.add_argument("--network_dim", type=int, default=16)
    p.add_argument("--network_alpha", type=float, default=8.0)
    p.add_argument("--network_args", type=str, nargs="*", default=[])
    p.add_argument("--optimizer_type", default="AdamW")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    # run knobs
    p.add_argument("--dit", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default=None)
    args = p.parse_args(argv)
    if args.auto_budget:
        args.compile = True  # activation_memory_budget only does anything with compile

    run_dir = make_run_dir("speed", label=args.label or "autobatch")
    cell_dir = run_dir / "cells"
    cell_dir.mkdir(exist_ok=True)
    mode = (f"auto-budget [{args.min_budget}..0.99]" if args.auto_budget
            else f"base_swap={args.blocks_to_swap} max_swap={args.max_swap} "
                 f"budget={args.activation_memory_budget}")
    print(f"auto-batch search: res={args.res} max_batch={args.max_batch} "
          f"grad_ckpt_res={args.gradient_checkpointing_resolutions or []} {mode} "
          f"compile={args.compile} net={args.network_module} opt={args.optimizer_type}", flush=True)

    frontier = []
    for res in args.res:
        found = (_search_budget if args.auto_budget else _search)(args, res, cell_dir, extra)
        if found is None:
            print(f"res {res}: OOM even at batch 1", flush=True)
            frontier.append({"res": res, "max_batch": 0, "oom": True})
        else:
            b, swap, budget, rec = found
            frontier.append({
                "res": res, "max_batch": b, "blocks_to_swap": swap, "budget": budget,
                "median_s_per_it": rec.get("median_s_per_it"),
                "it_per_s": rec.get("it_per_s"),
                "peak_reserved_mib": rec.get("peak_reserved_mib"),
                "grad_ckpt": rec.get("grad_ckpt"),
            })

    # All trials gathered in one place (success / OOM), so you don't scroll back.
    if _TRIAL_LOG:
        ok = sum(1 for line in _TRIAL_LOG if "[성공]" in line)
        print(f"\n{'=' * 56}\n전체 시도 요약 — 성공 {ok} / OOM {len(_TRIAL_LOG) - ok} "
              f"(총 {len(_TRIAL_LOG)})\n{'=' * 56}")
        for line in _TRIAL_LOG:
            print(line)

    print(f"\n{'=' * 56}\nMAX FEASIBLE BATCH PER RESOLUTION\n{'=' * 56}")
    for f in frontier:
        if f.get("oom"):
            print(f"  {f['res']:>4}:  OOM even at batch 1")
            continue
        sit = f.get("median_s_per_it") or 0.0
        gib = (f.get("peak_reserved_mib") or 0) / 1024
        gc = " +gc" if f.get("grad_ckpt") else ""
        sw = f" swap{f['blocks_to_swap']}" if f.get("blocks_to_swap") else ""
        bd = f" budget{f['budget']}" if (f.get("budget") or 1.0) < 1.0 else ""
        imgs = (f['max_batch'] / sit) if sit else 0.0
        speed = f"  {sit:.3f} s/it, {imgs:.2f} img/s  peak {gib:.1f} GiB" if sit else ""
        print(f"  {f['res']:>4}:  max batch {f['max_batch']}{gc}{sw}{bd}{speed}")

    out = write_result(run_dir, script=__file__, args=args,
                       metrics={"frontier": frontier, "max_batch": args.max_batch},
                       label=args.label, device=args.device)
    print(f"\nresult → {out}")


if __name__ == "__main__":
    main()
