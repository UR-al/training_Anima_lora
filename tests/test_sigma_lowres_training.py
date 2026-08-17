from types import SimpleNamespace

import numpy as np
import pytest
import torch

from library.datasets.base import BaseDataset
from library.datasets.buckets import demote_bucket_for
from library.io.cache_names import demoted_latents_key
from library.runtime.noise import draw_flat_sigmas
from train import AnimaTrainer, _sigma_yarn_scope


def _args(**overrides):
    values = {
        "sigma_lowres": True,
        "sigma_lowres_route": "1024:896",
        "sigma_lowres_threshold": 0.5,
        "sigma_lowres_threshold_max": None,
        "sigma_lowres_route2": "1024:768",
        "sigma_lowres_threshold2": 0.65,
        "sigma_lowres_threshold2_max": 0.95,
        "sigma_lowres_span": None,
        "sigma_lowres_span2": None,
        "sigma_lowres_yarnsig": None,
        "max_train_steps": 100,
        "gradient_accumulation_steps": 1,
        "seed": 42,
        "timestep_sampling": "sigmoid",
        "sigmoid_scale": 1.3,
        "sigmoid_bias": 0.0,
        "t_min": None,
        "t_max": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_secondary_sigma_rule_has_priority_inside_window() -> None:
    args = _args()
    assert AnimaTrainer._sigma_demote_choice(args, torch.tensor([0.8]), 1) == 2
    assert AnimaTrainer._sigma_demote_choice(args, torch.tensor([0.6]), 1) == 1
    assert AnimaTrainer._sigma_demote_choice(args, torch.tensor([0.4]), 1) is None


def test_secondary_route_requires_a_complete_window() -> None:
    args = _args(sigma_lowres_threshold2_max=None)
    with pytest.raises(ValueError, match="requires both"):
        AnimaTrainer._validate_sigma_rules(args)


def test_paired_sigma_draw_is_generator_stable() -> None:
    args = _args()
    first = draw_flat_sigmas(
        args,
        2,
        128,
        126,
        "cpu",
        generator=torch.Generator().manual_seed(123),
    )
    second = draw_flat_sigmas(
        args,
        2,
        128,
        126,
        "cpu",
        generator=torch.Generator().manual_seed(123),
    )
    assert torch.equal(first, second)


def test_dataset_reads_demoted_sibling_from_native_npz(tmp_path) -> None:
    bucket = demote_bucket_for(1008, 1024, 1024, 896)
    assert bucket is not None
    key = demoted_latents_key(*bucket)
    npz_path = tmp_path / "sample_1008x1024_anima.npz"
    expected = np.ones((16, bucket[1] // 8, bucket[0] // 8), dtype=np.float32)
    np.savez(npz_path, **{key: expected})

    dataset = BaseDataset.__new__(BaseDataset)
    dataset._demote_npz_cache = None
    dataset._sigma_demote_warned = set()
    info = SimpleNamespace(
        latents_npz=str(npz_path),
        bucket_reso=(1008, 1024),
    )
    loaded = dataset._load_demoted_sibling(info, (1024, 896))

    assert loaded is not None
    assert tuple(loaded.shape) == expected.shape


def test_yarn_scope_never_leaks_to_later_forwards() -> None:
    model = SimpleNamespace()
    with _sigma_yarn_scope(model, (1.1, 1.1, 1.0, 4.0, 0.5)):
        assert model._sigma_lowres_yarn is not None
    assert model._sigma_lowres_yarn is None
