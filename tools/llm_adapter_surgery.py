#!/usr/bin/env python3
"""Attach or strip the ``llm_adapter.*`` slice of an Anima DiT checkpoint.

Anima checkpoints carry a Qwen3 ``llm_adapter`` alongside the Cosmos-style DiT.
This does checkpoint surgery on that slice:

  strip   drop every ``llm_adapter.*`` tensor → a bare Cosmos-Predict2-style DiT
  attach  copy ``llm_adapter.*`` from a donor checkpoint onto a base that lacks it

Key-prefix styles are matched automatically (``llm_adapter.``,
``net.llm_adapter.``, ``diffusion_model.llm_adapter.``, and the
``net.diffusion_model.`` combination) so a ComfyUI-``net.``-prefixed export and a
training-side checkpoint both work. Original metadata is preserved.

Usage:
    python tools/llm_adapter_surgery.py strip  anima.safetensors --out dit_only.safetensors
    python tools/llm_adapter_surgery.py attach cosmos2.safetensors --donor anima.safetensors --out merged.safetensors

Independent reimplementation (MIT) of the attach/strip-llm-adapter tools from
bluvoll/diffusion-pipe (GPL-3); written clean-room against this repo's
``library.anima.weights`` prefix convention, so no upstream code is carried.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

_PREFIXES = ("net.", "diffusion_model.")


def _normalize(key: str) -> str:
    """Drop the outer ``net.`` / ``diffusion_model.`` wrappers (model-internal name)."""
    out = key
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if out.startswith(p):
                out = out[len(p) :]
                changed = True
    return out


def _is_adapter(key: str) -> bool:
    return _normalize(key).startswith("llm_adapter.")


def _load(path: Path) -> tuple[dict, dict[str, str]]:
    tensors: dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, metadata


def _detect_prefix(state: dict) -> str:
    """The outer prefix the base checkpoint uses for model tensors."""
    keys = list(state)
    for p in ("net.diffusion_model.", "diffusion_model.", "net."):
        if any(k.startswith(p) for k in keys):
            return p
    return ""


def _save(state: dict, meta: dict, out: Path, force: bool) -> None:
    out = out.expanduser().resolve()
    if out.exists() and not force:
        raise FileExistsError(f"Output exists: {out} (use --force).")
    save_file(state, str(out), metadata=meta)
    logger.info("Saved %d tensors → %s", len(state), out)


def cmd_strip(args: argparse.Namespace) -> None:
    src = args.input.expanduser().resolve()
    state, meta = _load(src)
    kept = {k: v for k, v in state.items() if not _is_adapter(k)}
    removed = len(state) - len(kept)
    logger.info("strip: %d llm_adapter tensors / %d total", removed, len(state))
    if removed == 0 and not args.allow_empty:
        raise SystemExit("No llm_adapter keys found (use --allow-empty to proceed).")
    if args.dry_run:
        logger.info("Dry run — no file written.")
        return
    meta["ss_llm_adapter_removed"] = str(removed)
    _save(kept, meta, args.out or src.with_name(f"{src.stem}_no_llm_adapter.safetensors"),
          args.force)


def cmd_attach(args: argparse.Namespace) -> None:
    base_path = args.input.expanduser().resolve()
    donor_path = args.donor.expanduser().resolve()
    base, base_meta = _load(base_path)
    donor, _ = _load(donor_path)

    donor_adapter = {_normalize(k): v for k, v in donor.items() if _is_adapter(k)}
    if not donor_adapter:
        raise SystemExit(f"Donor has no llm_adapter keys: {donor_path}")

    prefix = _detect_prefix(base)
    logger.info("attach: base prefix %r, donor adapter keys %d", prefix, len(donor_adapter))

    merged = dict(base)
    added = replaced = skipped = 0
    for norm_key, tensor in donor_adapter.items():
        target = f"{prefix}{norm_key}" if prefix else norm_key
        if target in merged:
            if not args.replace_existing:
                skipped += 1
                continue
            replaced += 1
        else:
            added += 1
        merged[target] = tensor
    logger.info("attach: added=%d replaced=%d skipped=%d", added, replaced, skipped)

    if args.dry_run:
        logger.info("Dry run — no file written.")
        return
    base_meta["ss_llm_adapter_source"] = str(donor_path)
    base_meta["ss_llm_adapter_added"] = str(added + replaced)
    _save(merged, base_meta,
          args.out or base_path.with_name(f"{base_path.stem}_with_llm_adapter.safetensors"),
          args.force)


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach/strip the llm_adapter of an Anima checkpoint.")
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("strip", help="Remove llm_adapter.* from a checkpoint.")
    s.add_argument("input", type=Path, help="Checkpoint to strip.")
    s.add_argument("--out", type=Path, default=None, help="Output (default <input>_no_llm_adapter).")
    s.add_argument("--allow-empty", action="store_true", help="Don't fail if no adapter keys.")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_strip)

    a = sub.add_parser("attach", help="Copy llm_adapter.* from a donor onto a base.")
    a.add_argument("input", type=Path, help="Base checkpoint (lacking the adapter).")
    a.add_argument("--donor", type=Path, required=True, help="Donor checkpoint with llm_adapter.*")
    a.add_argument("--out", type=Path, default=None, help="Output (default <base>_with_llm_adapter).")
    a.add_argument("--replace-existing", action="store_true", help="Overwrite adapter keys already in base.")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--force", action="store_true")
    a.set_defaults(func=cmd_attach)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
