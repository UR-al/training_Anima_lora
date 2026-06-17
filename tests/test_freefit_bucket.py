# -*- coding: utf-8 -*-
"""Free-fit bucket solver (native-aspect resize into a tier's token band).

Pure / torch-free. Covers band membership, rope-cap, aspect minimality, near-zero
crop, and determinism. Ported from the upstream free-fit PR, minus the max-ratio
clamp (this fork uses the rope cap as the only bound on extreme aspect ratios).
"""

import pytest

from library.datasets import buckets as bk


def _patch_grid(wh, patch=16):
    w, h = wh
    return w // patch, h // patch


def test_band_for_edge_1024():
    assert bk.freefit_band_for_edge(1024) == (4032, 4200)


@pytest.mark.parametrize("edge", [512, 768, 896, 1024, 1280, 1536])
def test_token_count_in_band(edge):
    lo, hi = bk.freefit_band_for_edge(edge)
    for w, h in [(1920, 1080), (1080, 1920), (1000, 1000), (2400, 1000)]:
        W, H = bk.freefit_bucket(w, h, (lo, hi))
        wp, hp = _patch_grid((W, H))
        assert lo <= wp * hp <= hi, f"{edge}: {wp * hp} out of band [{lo},{hi}]"
        assert W % 16 == 0 and H % 16 == 0


def test_rope_cap_respected():
    lo, hi = bk.freefit_band_for_edge(1536)  # largest band
    for w, h in [(4000, 1000), (1000, 4000)]:
        W, H = bk.freefit_bucket(w, h, (lo, hi), rope_cap=256)
        wp, hp = _patch_grid((W, H))
        assert max(wp, hp) <= 256


def test_aspect_is_minimized_vs_snap_bucket():
    # A 3:2 image: free-fit should match the aspect at least as well as the nearest
    # discrete 1024-tier bucket, and resize with ~zero crop.
    w, h = 1500, 1000
    band = bk.freefit_band_for_edge(1024)
    W, H = bk.freefit_bucket(w, h, band)
    ff_err = abs(W / H - w / h)
    snap = bk._nearest_aspect_bucket(w, h, bk.CONSTANT_TOKEN_BUCKETS)
    snap_err = abs(snap[0] / snap[1] - w / h)
    assert ff_err <= snap_err + 1e-9


def test_near_zero_crop():
    # Cover-resize to the free-fit grid then crop: residual on the covering axis
    # is strictly sub-patch (< 16 px) → crop is negligible.
    w, h = 1333, 1000
    W, H = bk.freefit_bucket(w, h, bk.freefit_band_for_edge(1024))
    scale = max(W / w, H / h)
    crop_px = max(round(w * scale) - W, round(h * scale) - H)
    assert crop_px < 16


def test_deterministic():
    band = bk.freefit_band_for_edge(1024)
    a = bk.freefit_bucket(1234, 1000, band)
    b = bk.freefit_bucket(1234, 1000, band)
    assert a == b


def test_invalid_band_raises():
    with pytest.raises(ValueError):
        bk.freefit_bucket(1000, 1000, (0, 10))
    with pytest.raises(ValueError):
        bk.freefit_bucket(1000, 1000, (4200, 4032))  # hi < lo


def test_square_image_square_grid():
    W, H = bk.freefit_bucket(1024, 1024, bk.freefit_band_for_edge(1024))
    wp, hp = _patch_grid((W, H))
    assert abs(wp - hp) <= 1  # ~square within one patch
    assert 4032 <= wp * hp <= 4200


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
