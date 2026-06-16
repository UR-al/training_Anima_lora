"""Per-subset ``is_val``: a subset flagged ``is_val = true`` is held out ENTIRELY
for validation (every image validates, none trains) — symmetric to ``is_reg``
(train-only). Validation in this trainer is otherwise a dataset-block split; this
flag lets the GUI's per-subset "Validation set (hold out)" toggle work.

Guards the wiring across three layers: the config schema must keep ``is_val``
(not silently drop it), ``DreamBoothSubsetParams`` must carry it, and
``generate_dataset_group_by_blueprint`` must route the subset's images to the
validation group and exclude its (empty) training block.
"""

import logging
import os

import pytest
from PIL import Image


def _make_subset(root: str, name: str, n: int) -> str:
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (512, 512), (i * 9, i * 9, i * 9)).save(
            os.path.join(d, f"{name}_{i}.png")
        )
        with open(os.path.join(d, f"{name}_{i}.txt"), "w", encoding="utf-8") as f:
            f.write("a photo of a thing")
    return d


def _blueprint(user_config: dict):
    import train
    from library.config.loader import BlueprintGenerator, ConfigSanitizer

    args = train.setup_parser().parse_args([])
    gen = BlueprintGenerator(ConfigSanitizer(support_dropout=True))
    return gen.generate(user_config, args)


def test_is_val_flag_survives_the_config_schema(tmp_path):
    """The sanitizer/blueprint must keep is_val on the subset param (a non-whitelisted
    key would be silently dropped → the feature would no-op)."""
    train_dir = _make_subset(str(tmp_path), "train", 2)
    val_dir = _make_subset(str(tmp_path), "valhold", 2)
    bp = _blueprint(
        {
            "datasets": [
                {"batch_size": 1, "subsets": [{"image_dir": train_dir}]},
                {
                    "batch_size": 1,
                    "subsets": [{"image_dir": val_dir, "is_val": True}],
                },
            ]
        }
    )
    flags = {
        os.path.basename(sb.params.image_dir): sb.params.is_val
        for db in bp.dataset_group.datasets
        for sb in db.subsets
    }
    assert flags == {"train": False, "valhold": True}


def test_is_val_subset_routes_whole_subset_to_validation(tmp_path):
    """is_val subset → all its images validate, none train; its (empty) training
    block is dropped, and a validation dataset is created for it."""
    logging.disable(logging.CRITICAL)
    try:
        from library.config.loader import generate_dataset_group_by_blueprint

        train_dir = _make_subset(str(tmp_path), "train", 3)
        val_dir = _make_subset(str(tmp_path), "valhold", 2)
        bp = _blueprint(
            {
                "datasets": [
                    {"batch_size": 1, "subsets": [{"image_dir": train_dir}]},
                    {
                        "batch_size": 1,
                        "subsets": [{"image_dir": val_dir, "is_val": True}],
                    },
                ]
            }
        )
        train_group, val_group = generate_dataset_group_by_blueprint(
            bp.dataset_group, constant_token_buckets=True
        )
        n_train = sum(d.num_train_images for d in train_group.datasets)
        n_val = sum(d.num_train_images for d in val_group.datasets) if val_group else 0
        assert n_train == 3, (
            f"training pool should exclude the is_val subset, got {n_train}"
        )
        assert n_val == 2, (
            f"validation pool should be the whole is_val subset, got {n_val}"
        )
        # the all-is_val block must NOT appear as a training dataset
        assert len(train_group.datasets) == 1
    finally:
        logging.disable(logging.NOTSET)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
