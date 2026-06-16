#!/usr/bin/env python3
"""Strip layer types from an Anima LoRA safetensors by substring match.

Drop every tensor whose key contains any of the given ``--strip`` substrings —
e.g. ablate the MLP adapters, keep only attention, or remove the ``llm_adapter``
slice from a LoRA. The original training metadata (``ss_*``) is preserved and a
small provenance note is appended.

Layer-type substrings in this repo's LoRA keys (use ``--list-types`` to see the
ones actually present in a file):
  self_attn  cross_attn  mlp  adaln_modulation  llm_adapter  final_layer

Usage:
    python tools/strip_lora_layers.py in.safetensors out.safetensors --strip mlp
    python tools/strip_lora_layers.py in.safetensors out.safetensors --strip llm_adapter cross_attn
    python tools/strip_lora_layers.py in.safetensors --list-types

Independent reimplementation (MIT) of the strip-lora-layers idea from
bluvoll/diffusion-pipe (GPL-3); written clean-room against this repo's key
naming + safetensors helpers, so no upstream code is carried.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Coarse module-type token taken from a flattened LoRA key (kohya names join the
# module path with underscores), used only for the --list-types overview.
_TYPE_TOKEN = re.compile(
    r"(self_attn|cross_attn|mlp|adaln_modulation|adaln|llm_adapter|"
    r"final_layer|embedder|modulation|norm)"
)


def _load(path: Path) -> tuple[dict, dict[str, str]]:
    tensors: dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = dict(f.metadata() or {})
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, metadata


def _list_types(state: dict) -> None:
    counts: dict[str, int] = {}
    for k in state:
        for tok in _TYPE_TOKEN.findall(k):
            counts[tok] = counts.get(tok, 0) + 1
    if not counts:
        logger.info("No recognized layer-type tokens found in %d keys.", len(state))
        return
    logger.info("Layer-type tokens present (token: #keys):")
    for tok, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        logger.info("  %-18s %d", tok, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Strip layer types from an Anima LoRA.")
    ap.add_argument("input", type=Path, help="Input LoRA .safetensors")
    ap.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output .safetensors (omit with --list-types)",
    )
    ap.add_argument(
        "--strip",
        nargs="+",
        metavar="SUBSTR",
        help="Key substrings to drop (e.g. mlp self_attn llm_adapter).",
    )
    ap.add_argument(
        "--list-types",
        action="store_true",
        help="List the layer-type tokens present in the file and exit.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Report what would be dropped; no write."
    )
    ap.add_argument(
        "--force", action="store_true", help="Overwrite output if it exists."
    )
    args = ap.parse_args()

    in_path = args.input.expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    state, metadata = _load(in_path)
    logger.info("Loaded %d tensors from %s", len(state), in_path.name)

    if args.list_types:
        _list_types(state)
        return

    if not args.strip:
        raise SystemExit("Nothing to do: pass --strip <substr...> or --list-types.")
    if args.output is None and not args.dry_run:
        raise SystemExit("Output path required (or use --dry-run / --list-types).")

    kept = {k: v for k, v in state.items() if not any(s in k for s in args.strip)}
    dropped = [k for k in state if k not in kept]
    logger.info(
        "Stripping %r → drop %d / keep %d", args.strip, len(dropped), len(kept)
    )
    for k in dropped[:10]:
        logger.info("  - %s", k)
    if len(dropped) > 10:
        logger.info("  ... and %d more", len(dropped) - 10)

    if not dropped:
        logger.warning("No keys matched %r — output would be identical.", args.strip)
    if args.dry_run:
        logger.info("Dry run — no file written.")
        return

    out_path = args.output.expanduser().resolve()
    if out_path.exists() and not args.force:
        raise FileExistsError(f"Output exists: {out_path} (use --force).")

    metadata["ss_stripped_layers"] = ",".join(args.strip)
    metadata["ss_stripped_count"] = str(len(dropped))
    save_file(kept, str(out_path), metadata=metadata)
    logger.info("Saved %d tensors → %s", len(kept), out_path)


if __name__ == "__main__":
    main()
