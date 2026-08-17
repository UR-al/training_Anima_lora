from pathlib import Path

import numpy as np
import torch
from PIL import Image

from library.datasets.buckets import demote_bucket_for, demoted_token_counts
from library.io.cache_names import demoted_latents_key
from library.preprocess.latents import cache_demoted_latents, get_latents_npz_path


class _DummyVAE:
    device = torch.device("cpu")
    dtype = torch.float32

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = pixels.shape
        return torch.zeros(batch, 16, height // 8, width // 8)


def test_demote_bucket_only_accepts_native_route_band() -> None:
    demoted = demote_bucket_for(1008, 1024, 1024, 896)
    assert demoted is not None
    assert demote_bucket_for(720, 768, 1024, 896) is None
    assert demoted_token_counts({(1008, 1024)}, 1024, 896) == {
        (demoted[0] // 16) * (demoted[1] // 16)
    }


def test_demoted_latent_is_appended_inside_native_npz(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (1008, 1024), "white").save(image_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    native_npz = get_latents_npz_path(
        image_path,
        (1008, 1024),
        cache_dir=cache_dir,
        image_dir=tmp_path,
    )
    np.savez(native_npz, latents_128x126=np.ones((16, 128, 126), dtype=np.float32))

    stats = cache_demoted_latents(
        tmp_path,
        _DummyVAE(),
        native_edge=1024,
        demote_edge=896,
        cache_dir=cache_dir,
        batch_size=1,
    )

    bucket = demote_bucket_for(1008, 1024, 1024, 896)
    assert bucket is not None
    key = demoted_latents_key(*bucket)
    assert stats.written == 1
    with np.load(native_npz) as cached:
        assert "latents_128x126" in cached
        assert key in cached
        assert cached[key].shape == (16, bucket[1] // 8, bucket[0] // 8)
