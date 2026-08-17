#!/usr/bin/env python3
"""Cache VAE latents for all images in a dataset directory.

Encodes images through the Qwen Image VAE and saves latent caches (.npz)
alongside the images (or under ``--cache_dir``).  Skips already-cached
entries (idempotent).

The walk → group-by-resolution → encode → save loop lives in
``library/preprocess/latents.py``; this file is argparse + VAE load + reporting.
"""

import argparse
from pathlib import Path

import torch


from library.preprocess import cache_demoted_latents, cache_latents, tqdm_progress
from library.runtime.cli import add_io_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(
        parser,
        cache_noun="latent caches",
        include_batch_size=True,
        batch_size_default=4,
    )
    parser.add_argument("--vae", type=str, required=True, help="Path to VAE weights")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=64,
        help="VAE spatial chunk size (default: 64)",
    )
    parser.add_argument(
        "--disable_cache",
        action="store_true",
        default=True,
        help="Disable VAE internal cache (default: True)",
    )
    parser.add_argument(
        "--qwen_image_vae_2d",
        action="store_true",
        help="Use the image-only 2D Qwen-Image VAE for encoding (~2x faster, ~1/3 VRAM; "
        "latents numerically equivalent to the 3D VAE for single images, so caches stay "
        "valid). Recommended for this caching step.",
    )
    parser.add_argument(
        "--path_pattern",
        "--path-pattern",
        dest="path_pattern",
        default="*",
        help=(
            "Only cache images whose path relative to --dir matches this "
            "fnmatch glob. Use | to separate alternatives. Default: *"
        ),
    )
    parser.add_argument(
        "--sigma_demote",
        metavar="NATIVE:DEMOTE",
        help=(
            "Append a lower-resolution latent sibling to each eligible native "
            "NPZ instead of rebuilding native latents, for example 1024:896."
        ),
    )
    args = parser.parse_args()

    demote_route = None
    if args.sigma_demote:
        native_s, separator, demote_s = args.sigma_demote.partition(":")
        try:
            if not separator:
                raise ValueError
            demote_route = (int(native_s), int(demote_s))
        except ValueError:
            raise SystemExit(
                "--sigma_demote expects one NATIVE:DEMOTE route, for example "
                f"1024:896; got {args.sigma_demote!r}"
            ) from None
        if not demote_route[0] > demote_route[1] > 0:
            raise SystemExit(
                "--sigma_demote requires NATIVE > DEMOTE > 0; got "
                f"{args.sigma_demote!r}"
            )

    from library.models import qwen_vae as qwen_image_autoencoder_kl

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading VAE from {args.vae} ...")
    vae = qwen_image_autoencoder_kl.load_vae_2d_or_3d(
        args.vae,
        use_2d=args.qwen_image_vae_2d,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.chunk_size,
        disable_cache=args.disable_cache,
    )
    vae.to(device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()

    if demote_route is None:
        stats = cache_latents(
            data_dir,
            vae,
            cache_dir=cache_dir,
            recursive=args.recursive,
            path_pattern=args.path_pattern,
            batch_size=args.batch_size,
            progress=tqdm_progress("Caching latents"),
        )
    else:
        stats = cache_demoted_latents(
            data_dir,
            vae,
            native_edge=demote_route[0],
            demote_edge=demote_route[1],
            cache_dir=cache_dir,
            recursive=args.recursive,
            path_pattern=args.path_pattern,
            batch_size=args.batch_size,
            progress=tqdm_progress(
                f"Caching demoted latents {demote_route[0]}:{demote_route[1]}"
            ),
        )
    print(
        f"\nLatent caching complete: {stats.written} cached, "
        f"{stats.skipped} skipped (already existed)"
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
