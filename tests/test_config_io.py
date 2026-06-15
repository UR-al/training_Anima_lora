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
timestep_sample_method = "sigmoid"
sigmoid_scale = 1.3
weighting_scheme = "logit_normal"
attn_mode = "flash"
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


def test_load_applies_lets_renames_and_drops():
    ef = load_toml_to_form(_LETS)["extra_flags"]
    assert "--timestep_sampling sigmoid" in ef       # timestep_sample_method →
    assert "--use_vae_cache" in ef                   # cache_latents →
    assert "--output_config" in ef                   # save_toml →
    assert "--t_min 0.0" in ef and "--t_max 1.0" in ef  # 0/1000 ÷1000 → σ∈[0,1]
    assert "--sigmoid_scale 1.3" in ef
    assert "--loss_type huber" in ef and "--huber_c 0.1" in ef
    # kohya AR-bucketing keys have no anima_lora equivalent → dropped.
    assert "enable_bucket" not in ef and "max_bucket_reso" not in ef


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
    ef = back.get("extra_flags", "")
    assert "--huber_c 0.1" in ef
    assert "--no-masked_loss" in ef  # false bool preserved as a negation


if __name__ == "__main__":  # allow `python tests/test_config_io.py`
    test_load_maps_dedicated_fields()
    test_load_applies_lets_renames_and_drops()
    test_save_emits_runnable_toml_and_round_trips()
    print("all config_io round-trip tests passed")
