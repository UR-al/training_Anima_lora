#!/usr/bin/env python
"""XYZ-plot-style grid explorer: find the max feasible training config + speed.

Fix the optimizer + network, then sweep the memory levers as independent axes —
``activation_memory_budget``, ``blocks_to_swap``, ``batch``, ``gradient_checkpointing``,
``compile`` — across resolution tiers, and for EVERY combination report whether it
OOMs and, if it fits, the per-step speed over ~20 steps. The point is to read off
the limit: the largest (resolution, batch) your card can hold and the cheapest /
fastest setting that gets there.

Standalone — nothing to do with the GUI. NO images / preprocessing needed: VRAM and
speed are determined by tensor SHAPES (batch, token count, channels) + the config,
not by pixel values, so synthetic latents give the same OOM frontier and timing as
real cached latents. Each cell runs in its OWN subprocess (run_bench.py) so a fresh
CUDA allocator measures each config cleanly — fragmentation/caching across configs
can't poison the OOM boundary, and an uncatchable OOM only kills that one cell.

Axes accept explicit values OR a ``range:`` shorthand:
    --blocks-to-swap range:0-26:2      → 0,2,4,…,26
    --budget range:0.1-0.9:0.2         → 0.1,0.3,0.5,0.7,0.9
    --grad-ckpt on off                 → both
    --res 1024 1536  --batch 1 2 4

Anything after ``--`` is EXTRA, forwarded verbatim to every cell (e.g. a fixed
``--network_alpha 8`` or ``--optimizer_args weight_decay=0.01``).

Dependency the sweep encodes for you: ``activation_memory_budget`` only does
anything with ``--compile on`` AND ``--grad-ckpt off`` (it's a torch.compile
partitioner knob, ignored under gradient checkpointing). For other cells the budget
axis collapses to 1.0 so you don't pay for redundant runs.

Examples:
    # the full "how far can I push it" sweep (coarse, ~manageable)
    python tasks.py bench-sweep --res 1024 1536 --batch 1 2 \
        --grad-ckpt on off --blocks-to-swap range:0-26:4 --label limits
    # budget sweep (needs compile, no grad-ckpt)
    python tasks.py bench-sweep --res 1536 --batch 1 2 --compile on --grad-ckpt off \
        --budget range:0.1-0.9:0.1 --blocks-to-swap 0 8 --label budget

Output: a per-cell CSV (the raw XYZ grid) + a feasibility-frontier summary on the
console + a master result.json, all under bench/speed/results/<ts>[-<label>]/.
Use --dry-run to print the cell list + ETA without running anything.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

from bench._common import REPO_ROOT, make_run_dir, write_result

RUN_BENCH = Path(__file__).resolve().parent / "run_bench.py"
_TRUE = {"on", "1", "true", "yes"}
_FALSE = {"off", "0", "false", "no"}


def _parse_range(tok: str, cast):
    """`range:start-stop[:step]` → inclusive list. Float step keeps fp precision."""
    body = tok[len("range:") :]
    step = "1"
    if ":" in body:
        body, step = body.split(":", 1)
    start_s, stop_s = body.split("-", 1)
    start, stop, st = cast(start_s), cast(stop_s), cast(step)
    out = []
    # integer count to avoid fp drift accumulating past stop
    n = int(round((stop - start) / st)) if st else 0
    for i in range(n + 1):
        out.append(cast(round(start + i * st, 6)) if cast is float else start + i * st)
    return out


def _axis(tokens, cast):
    vals = []
    for t in tokens:
        if isinstance(t, str) and t.startswith("range:"):
            vals.extend(_parse_range(t, cast))
        else:
            vals.append(cast(t))
    # dedupe, preserve order
    seen, uniq = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _bool_axis(tokens):
    out = []
    for t in tokens:
        s = str(t).lower()
        if s in _TRUE:
            out.append(True)
        elif s in _FALSE:
            out.append(False)
        else:
            raise SystemExit(f"on/off expected, got {t!r}")
    seen, uniq = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def main() -> None:
    # argv split on the first bare "--": ours | EXTRA forwarded to each cell.
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--res", nargs="+", default=["512", "1024", "1536"], help="Resolution edges.")
    p.add_argument("--batch", nargs="+", default=["1", "2"], help="Batch sizes (range: ok).")
    p.add_argument("--grad-ckpt", dest="grad_ckpt", nargs="+", default=["on", "off"], help="on/off (both default).")
    p.add_argument("--compile", nargs="+", default=["off"], help="on/off torch.compile.")
    p.add_argument("--blocks-to-swap", dest="swap", nargs="+", default=["0"], help="0..26 (range: ok).")
    p.add_argument("--budget", nargs="+", default=["1.0"], help="0.1..0.99 (range: ok; needs compile+no-gradckpt).")
    # fixed config (forwarded to every cell)
    p.add_argument("--dit", default=None)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--network_dim", type=int, default=16)
    p.add_argument("--network_alpha", type=float, default=8.0)
    p.add_argument("--optimizer_type", default="AdamW")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=20, help="Timed steps when a cell fits.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--compile_mode", default=None)
    p.add_argument("--compile_dynamic_seq", action="store_true")
    p.add_argument("--unsloth_offload_checkpointing", action="store_true")
    p.add_argument("--cpu_offload_checkpointing", action="store_true")
    # sweep control
    p.add_argument("--label", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="List cells + ETA, run nothing.")
    p.add_argument("--yes", action="store_true", help="Skip the large-grid confirmation.")
    p.add_argument("--max-cells", dest="max_cells", type=int, default=120, help="Confirm above this many cells.")
    args = p.parse_args(argv)

    res = _axis(args.res, int)
    batch = _axis(args.batch, int)
    grad_ckpt = _bool_axis(args.grad_ckpt)
    compile_ax = _bool_axis(args.compile)
    swap = _axis(args.swap, int)
    budget = _axis(args.budget, float)

    # A cell is one model build = (grad_ckpt, compile, swap, budget, res); batch is
    # swept INSIDE each cell (cheap, same model). budget only matters with
    # compile+no-gradckpt — collapse it to 1.0 elsewhere so we don't rebuild for a
    # no-op knob.
    cells = []
    for gc, cp, sw, res_edge in itertools.product(grad_ckpt, compile_ax, swap, res):
        budgets = budget if (cp and not gc) else [1.0]
        for bud in budgets:
            cells.append({"grad_ckpt": gc, "compile": cp, "swap": sw, "budget": bud, "res": res_edge})
    # dedupe (collapsed budgets can repeat)
    seen, uniq = set(), []
    for c in cells:
        key = (c["grad_ckpt"], c["compile"], c["swap"], c["budget"], c["res"])
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    cells = uniq

    per_cell_s = 25 + (45 if any(c["compile"] for c in cells) else 0) + args.steps * 1.0
    eta_min = len(cells) * per_cell_s / 60.0
    print(f"grid: {len(cells)} cells × batch{batch} "
          f"(res={res} grad_ckpt={grad_ckpt} compile={compile_ax} swap={swap} budget={budget})")
    print(f"rough ETA: ~{eta_min:.0f} min ({len(cells)} model builds; each loads the DiT)")

    if args.dry_run:
        for c in cells:
            print("  cell:", c)
        return
    if len(cells) > args.max_cells and not args.yes:
        raise SystemExit(
            f"{len(cells)} cells > --max-cells {args.max_cells}. Re-run with --yes to proceed, "
            "or coarsen the axes (e.g. --blocks-to-swap range:0-26:4)."
        )

    run_dir = make_run_dir("speed", label=args.label or "sweep")
    cell_dir = run_dir / "cells"
    cell_dir.mkdir(exist_ok=True)

    rows = []  # one per (cell, batch)
    t_start = time.time()
    for idx, c in enumerate(cells):
        out_json = cell_dir / f"cell_{idx:04d}.json"
        cmd = [
            sys.executable, str(RUN_BENCH),
            "--device", args.device, "--dtype", args.dtype, "--attn_mode", args.attn_mode,
            "--seed", str(args.seed),
            "--tiers", str(c["res"]),
            "--batch", *[str(b) for b in batch],
            "--steps", str(args.steps), "--warmup", str(args.warmup),
            "--blocks_to_swap", str(c["swap"]),
            "--network_dim", str(args.network_dim), "--network_alpha", str(args.network_alpha),
            "--optimizer_type", args.optimizer_type, "--learning_rate", str(args.learning_rate),
            "--activation_memory_budget", str(c["budget"]),
            "--out_json", str(out_json),
            "--label", f"cell{idx:04d}",
        ]
        if args.dit:  # else run_bench resolves the DiT from the config chain
            cmd += ["--dit", args.dit]
        if c["grad_ckpt"]:
            cmd.append("--gradient_checkpointing")
            if args.unsloth_offload_checkpointing:
                cmd.append("--unsloth_offload_checkpointing")
            elif args.cpu_offload_checkpointing:
                cmd.append("--cpu_offload_checkpointing")
        if c["compile"]:
            cmd.append("--compile")
            if args.compile_mode:
                cmd += ["--compile_mode", args.compile_mode]
            if args.compile_dynamic_seq:
                cmd.append("--compile_dynamic_seq")
        cmd += extra

        tag = (f"gc={'Y' if c['grad_ckpt'] else 'N'} cmp={'Y' if c['compile'] else 'N'} "
               f"swap={c['swap']:>2} bud={c['budget']:.2g} res={c['res']}")
        print(f"[{idx + 1}/{len(cells)}] {tag} …", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))

        cell_runs = []
        if out_json.exists():
            try:
                cell_runs = json.loads(out_json.read_text()).get("runs", [])
            except Exception:  # noqa: BLE001
                cell_runs = []
        got = {(r.get("tier"), r.get("batch")) for r in cell_runs}
        for r in cell_runs:
            rows.append({**c, **r})
        # res is fixed per cell; any batch with no record → the process died on it
        # (fatal/uncatchable OOM) before recording → mark as OOM(crash).
        for b in batch:
            if (c["res"], b) not in got:
                rows.append({**c, "tier": c["res"], "batch": b, "tokens": None,
                             "oom": True, "crashed": proc.returncode != 0})

    _report(args, run_dir, cells, batch, rows, time.time() - t_start)


def _report(args, run_dir, cells, batch, rows, elapsed):
    # CSV: the raw XYZ grid.
    cols = ["res", "batch", "tokens", "grad_ckpt", "compile", "swap", "budget",
            "oom", "median_s_per_it", "it_per_s", "peak_reserved_mib"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    (run_dir / "sweep.csv").write_text("\n".join(lines), encoding="utf-8")

    feasible = [r for r in rows if not r.get("oom")]
    print(f"\n{'=' * 64}\nFEASIBILITY FRONTIER  ({len(feasible)}/{len(rows)} cells fit)  "
          f"[{elapsed / 60:.1f} min]\n{'=' * 64}")
    # Per (res,batch): the fastest feasible config (the practical pick).
    by_target: dict = {}
    for r in rows:
        by_target.setdefault((r["res"], r["batch"]), []).append(r)
    for (res_edge, b) in sorted(by_target):
        rs = by_target[(res_edge, b)]
        fit = [r for r in rs if not r.get("oom")]
        if not fit:
            print(f"  res {res_edge:>4} batch {b}:  OOM in all {len(rs)} configs")
            continue
        best = min(fit, key=lambda r: r.get("median_s_per_it") or 9e9)
        cfg = (f"gc={'Y' if best['grad_ckpt'] else 'N'} swap={best['swap']} "
               f"bud={best['budget']:.2g}" + (" compile" if best["compile"] else ""))
        sit = best.get("median_s_per_it")
        gib = (best.get("peak_reserved_mib") or 0) / 1024
        print(f"  res {res_edge:>4} batch {b}:  fits ({len(fit)}/{len(rs)})  "
              f"fastest {sit:.3f} s/it ({1 / sit:.2f} it/s) @ {cfg}  peak {gib:.1f} GiB")
    # Headline: the largest (res, batch) that fits at all.
    fits_targets = {(r["res"], r["batch"]) for r in feasible}
    if fits_targets:
        mx = max(fits_targets, key=lambda t: (t[0], t[1]))
        print(f"\nMAX FEASIBLE (res, batch): {mx[0]} @ batch {mx[1]}")

    metrics = {"cells": len(cells), "rows": len(rows), "feasible": len(feasible),
               "elapsed_min": round(elapsed / 60, 2), "grid": rows}
    out = write_result(run_dir, script=__file__, args=args, metrics=metrics,
                       label=args.label, device=args.device)
    print(f"\ncsv  → {run_dir / 'sweep.csv'}\njson → {out}")


if __name__ == "__main__":
    main()
