"""Auto-captioning tasks — Qwen vision-language captioner.

``tasks.py qwen-caption`` captions a set of images (a ``--manifest`` file of paths, or
``--paths a.png b.png``) with the model configured in ``dataset_tags/qwen_caption.toml``.
The Dataset tab spawns this as a subprocess over the selected images so torch never
loads in the GUI process. Caption/prompt logic lives in
``library.captioning.qwen_caption``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ._common import ROOT


def cmd_qwen_caption(extra):
    """Caption images with the configured Qwen VLM (writes <stem>.txt)."""
    from library.captioning import qwen_caption as qc

    ap = argparse.ArgumentParser(prog="tasks.py qwen-caption")
    ap.add_argument(
        "--manifest", help="text file with one image path per line (# comments ok)"
    )
    ap.add_argument("--paths", nargs="*", default=[], help="image paths (inline)")
    ap.add_argument("--mode", choices=qc.VALID_MODES, default="tags")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="re-caption images that already have a non-empty .txt",
    )
    ap.add_argument(
        "--config",
        default=str(ROOT / qc.DEFAULT_CONFIG_REL),
        help="captioner TOML (default: dataset_tags/qwen_caption.toml)",
    )
    args = ap.parse_args(extra)

    paths: list[str] = list(args.paths)
    if args.manifest:
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                paths.append(line)
    if not paths:
        raise SystemExit("no images given (use --manifest or --paths)")

    try:
        cfg = qc.load_caption_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nEdit dataset_tags/qwen_caption.toml and set model_path."
        ) from exc

    print(
        f"Qwen caption: {len(paths)} image(s), mode={args.mode}, "
        f"loader={cfg.get('loader')}, model={cfg.get('model_path')}"
    )
    qc.caption_paths(paths, args.mode, args.overwrite, cfg)
