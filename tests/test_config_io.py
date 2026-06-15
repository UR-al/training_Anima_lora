# -*- coding: utf-8 -*-
"""Round-trip tests for the GUI config save/load (gui.modules.config_io).

Pure TOML/dict logic — no torch, no Gradio — so it runs anywhere. Anchored on a
real LoRA_Easy_Training_Scripts (LETS) LoKr + CAME export to guard the key
renames and the form mapping the kohya GUI's Load/Save buttons depend on.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.modules.config_io import load_toml_to_form, save_form_to_toml  # noqa: E402

# A representative slice of the user's real LETS config (flat keys only).
_LETS = """
train_mode = "lora"
seed = 42
max_train_epochs = 50
mixed_precision = "bf16"
gradient_checkpointing = true
cache_latents = true
network_dim = 100000
network_alpha = 1
network_train_unet_only = true
network_args = ["algo=lokr", "preset=unet-transformer-only", "factor=4", "full_matrix=True"]
min_timestep = 0
max_timestep = 1000
lr_scheduler = "constant"
optimizer_type = "LoraEasyCustomOptimizer.came.CAME"
learning_rate = 8e-06
optimizer_args = ["weight_decay=0.0001", "update_strategy=cautious"]
loss_type = "huber"
huber_c = 0.1
save_toml = true
save_state = true
timestep_sample_method = "sigmoid"
sigmoid_scale = 1.3
weighting_scheme = "logit_normal"
attn_mode = "flash"
qwen3_max_token_length = 512
vae_batch_size = 1
enable_bucket = true
max_bucket_reso = 4096
pretrained_model_name_or_path = "C:/anima.safetensors"
vae = "C:/vae.safetensors"
qwen3 = "C:/te.safetensors"
"""


def test_load_maps_dedicated_fields():
    form = load_toml_to_form(_LETS)
    assert form["optimizer_type"] == "LoraEasyCustomOptimizer.came.CAME"
    assert form["network_dim"] == "100000"
    assert form["network_alpha"] == "1"
    assert form["lr_scheduler_type"] == "constant"  # lr_scheduler → lr_scheduler_type
    assert form["dit_path"].endswith("anima.safetensors")  # model-path rename
    assert form["vae_path"].endswith("vae.safetensors")
    assert form["te_path"].endswith("te.safetensors")
    # LyCORIS algo/preset ride inside network_args verbatim, not extra_flags.
    assert "algo=lokr" in form["network_args"]
    assert "algo=lokr" not in form.get("extra_flags", "")


def test_load_renames_to_dedicated_fields():
    """LETS renames + training knobs land in dedicated form fields (Phase 1b)."""
    form = load_toml_to_form(_LETS)
    assert form["timestep_sampling"] == "sigmoid"  # timestep_sample_method →
    assert form["sigmoid_scale"] == "1.3"
    assert form["weighting_scheme"] == "logit_normal"
    assert form["loss_type"] == "huber" and form["huber_c"] == "0.1"
    assert form["attn_mode"] == "flash" and form["mixed_precision"] == "bf16"
    assert form["qwen3_max_token_length"] == "512"
    # 0/1000 ÷1000 → flow-matching σ, as dedicated t_min/t_max fields.
    assert form["t_min"] == "0.0" and form["t_max"] == "1.0"
    # bool renames → checkbox fields.
    assert form["use_vae_cache"] is True  # cache_latents →
    assert form["output_config"] is True  # save_toml →
    assert form["save_state"] is True
    assert form["gradient_checkpointing"] is True


def test_load_drops_and_extra_routing():
    form = load_toml_to_form(_LETS)
    ef = form.get("extra_flags", "")
    # kohya AR-bucketing keys have no anima_lora equivalent → dropped entirely.
    assert "enable_bucket" not in ef and "max_bucket_reso" not in ef
    # a key with no dedicated field still round-trips via Extra CLI flags.
    assert "--vae_batch_size 1" in ef


def test_save_emits_runnable_toml_and_round_trips():
    form = {
        "method": "lycoris",
        "preset": "default",
        "optimizer_type": "CAME",
        "learning_rate": "8e-06",
        "network_module": "networks.lycoris_anima",
        "network_dim": "100000",
        "network_alpha": "1",
        "network_args": "algo=lokr preset=unet-transformer-only factor=4 full_matrix=True",
        "optimizer_args": "weight_decay=0.0001 update_strategy=cautious",
        "extra_flags": "--loss_type huber --huber_c 0.1 --no-masked_loss --use_vae_cache",
    }
    text = save_form_to_toml(form)
    assert "python train.py --config_file" in text  # header hint

    back = load_toml_to_form(text)
    assert back["optimizer_type"] == "CAME"
    assert back["network_dim"] == "100000"
    assert "algo=lokr" in back["network_args"]
    # loss_type/huber_c/use_vae_cache are dedicated fields after Phase 1b.
    assert back["loss_type"] == "huber" and back["huber_c"] == "0.1"
    assert back["use_vae_cache"] is True
    # masked_loss has no dedicated field → round-trips via Extra CLI flags.
    assert "--no-masked_loss" in back.get("extra_flags", "")


_KOHYA_DATASET = """
[[datasets]]
resolution = 960
batch_size = 2
  [[datasets.subsets]]
  image_dir = "C:/data/char"
  num_repeats = 5
  keep_tokens = 1
  caption_extension = ".txt"
  flip_aug = true
"""

_ANIMA_DATASET = """
[[datasets]]
batch_size = 1
  [[datasets.subsets]]
  image_dir = "post_image_dataset/resized"
  cache_dir = "post_image_dataset/lora"
  tiers = [512, 1024]
  batch_size = 4
"""


def test_load_harvests_kohya_dataset_block():
    form = load_toml_to_form(_KOHYA_DATASET)
    assert form["ds_image_dir"] == "C:/data/char"
    assert form["ds_num_repeats"] == "5" and form["ds_keep_tokens"] == "1"
    assert form["ds_caption_extension"] == ".txt"
    assert form["ds_batch_size"] == "2"  # dataset-level batch_size
    assert form["ds_flip_aug"] is True
    assert form["target_res"] == ["896"]  # kohya resolution 960 → nearest tier
    # the [[datasets]] section never leaks into the flat router / extra_flags.
    assert "datasets" not in form.get("extra_flags", "")


def test_load_harvests_anima_dataset_tiers():
    form = load_toml_to_form(_ANIMA_DATASET)
    assert form["ds_cache_dir"] == "post_image_dataset/lora"
    assert form["ds_tiers"] == "512,1024"
    assert form["ds_batch_size"] == "4"  # subset batch_size wins over block
    assert "target_res" not in form  # no `resolution` key → no tier snap


def test_load_without_dataset_leaves_ds_fields_unset():
    form = load_toml_to_form('optimizer_type = "CAME"\n')
    assert not any(k.startswith("ds_") for k in form)


if __name__ == "__main__":  # allow `python tests/test_config_io.py`
    test_load_maps_dedicated_fields()
    test_load_renames_to_dedicated_fields()
    test_load_drops_and_extra_routing()
    test_save_emits_runnable_toml_and_round_trips()
    test_load_harvests_kohya_dataset_block()
    test_load_harvests_anima_dataset_tiers()
    test_load_without_dataset_leaves_ds_fields_unset()
    print("all config_io round-trip tests passed")
