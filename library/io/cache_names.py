"""Torch-free cache naming helpers shared by preprocess and datasets."""

from __future__ import annotations


def demoted_latents_key(width: int, height: int) -> str:
    """NPZ key for a sigma-lowres sibling latent at pixel size ``(W, H)``."""
    return f"demoted_{height // 8}x{width // 8}"
