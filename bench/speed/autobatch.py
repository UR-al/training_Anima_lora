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


def _trial(args, res, batch, swap, out_json, extra):
    """Run one (res, batch, swap) trial via run_bench; return (fits: bool, rec: dict).

    rec carries median_s_per_it / peak_reserved_mib / grad_ckpt when it fit. A
    missing out_json (fatal/uncatchable OOM crash) counts as not-fit.
    """
    cmd = [
        sys.executable, str(RUN_BENCH),
        "--device", args.device, "--dtype", args.dtype, "--attn_mode", args.attn_mode,
        "--seed", str(args.seed),
        "--plan", f"{res}:{batch}",
        "--steps", str(args.steps), "--warmup", str(args.warmup),
        "--blocks_to_swap", str(swap),
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
    cfg = (f"res={res} batch={batch} swap={swap}" + (" +gc" if gc else "")
           + f"  [{net} / {args.optimizer_type}]")
    gib = (rec.get("peak_reserved_mib") or 0) / 1024
    if fits:
        sit = rec.get("median_s_per_it") or 0.0
        speed = f"{sit:.3f} s/it ({1 / sit:.2f} it/s)" if sit else "측정됨"
        print(f"  [성공] {cfg}  ->  {speed}, peak {gib:.1f} GiB", flush=True)
    else:
        print(f"  [OOM ] {cfg}  ->  이 설정으로 OOM 났습니다 (peak {gib:.1f} GiB)", flush=True)
    return fits, rec


def _max_batch(args, res, swap, cell_dir, extra):
    """Binary-search the largest batch in [1, --max-batch] that fits at ``res`` with
    ``swap`` blocks swapped. Returns (batch, rec) of the best fit, or None if even
    batch 1 OOMs."""
    lo, hi, best = 1, args.max_batch, None
    while lo <= hi:
        mid = (lo + hi) // 2
        fits, rec = _trial(args, res, mid, swap, cell_dir / f"r{res}_s{swap}_b{mid}.json", extra)
        if fits:
            best = (mid, rec)
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _search(args, res, cell_dir, extra):
    """Find (batch, swap, rec) for this resolution. Max batch at the BASE swap
    (--blocks_to_swap); if even batch 1 OOMs there and --max-swap permits it,
    binary-search the MINIMAL blocks_to_swap that fits batch 1 (swap is slow → use
    the least needed; fit is monotonic in swap), then the max batch at that swap.
    Returns None if infeasible even at --max-swap."""
    base = args.blocks_to_swap
    best = _max_batch(args, res, base, cell_dir, extra)
    if best is not None:
        return best[0], base, best[1]
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
    return (best[0], fit_swap, best[1]) if best else (1, fit_swap, fit_rec)


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
    p.add_argument("--compile", action="store_true")
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

    run_dir = make_run_dir("speed", label=args.label or "autobatch")
    cell_dir = run_dir / "cells"
    cell_dir.mkdir(exist_ok=True)
    print(f"auto-batch search: res={args.res} max_batch={args.max_batch} "
          f"grad_ckpt_res={args.gradient_checkpointing_resolutions or []} "
          f"base_swap={args.blocks_to_swap} max_swap={args.max_swap} "
          f"net={args.network_module} opt={args.optimizer_type}", flush=True)

    frontier = []
    for res in args.res:
        found = _search(args, res, cell_dir, extra)
        if found is None:
            print(f"res {res}: OOM even at batch 1 (max swap {args.max_swap})", flush=True)
            frontier.append({"res": res, "max_batch": 0, "oom": True})
        else:
            b, swap, rec = found
            frontier.append({
                "res": res, "max_batch": b, "blocks_to_swap": swap,
                "median_s_per_it": rec.get("median_s_per_it"),
                "it_per_s": rec.get("it_per_s"),
                "peak_reserved_mib": rec.get("peak_reserved_mib"),
                "grad_ckpt": rec.get("grad_ckpt"),
            })

    print(f"\n{'=' * 56}\nMAX FEASIBLE BATCH PER RESOLUTION\n{'=' * 56}")
    for f in frontier:
        if f.get("oom"):
            print(f"  {f['res']:>4}:  OOM even at batch 1")
            continue
        sit = f.get("median_s_per_it") or 0.0
        gib = (f.get("peak_reserved_mib") or 0) / 1024
        gc = " +gc" if f.get("grad_ckpt") else ""
        sw = f" swap{f['blocks_to_swap']}" if f.get("blocks_to_swap") else ""
        speed = f"  {sit:.3f} s/it ({1 / sit:.2f} it/s)  peak {gib:.1f} GiB" if sit else ""
        print(f"  {f['res']:>4}:  max batch {f['max_batch']}{gc}{sw}{speed}")

    out = write_result(run_dir, script=__file__, args=args,
                       metrics={"frontier": frontier, "max_batch": args.max_batch},
                       label=args.label, device=args.device)
    print(f"\nresult → {out}")


if __name__ == "__main__":
    main()
