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
ip_noise_gamma = 0.1
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
    # vae_batch_size is a curated field now (kohya-parity promotion), not extra_flags.
    assert form["vae_batch_size"] == "1" and "vae_batch_size" not in ef
    # a still-uncurated key round-trips via Extra CLI flags.
    assert "--ip_noise_gamma 0.1" in ef


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
    # masked_loss is a curated tri-state now → --no-masked_loss round-trips to "off".
    assert back["masked_loss"] == "off"


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


def test_load_constantcosine_fields():
    form = load_toml_to_form(
        "use_constantcosine = true\n"
        "constantcosine_tail_epochs = 8\n"
        "lr_scheduler_min_lr_ratio = 0.05\n"
    )
    assert form["use_constantcosine"] is True
    assert form["constantcosine_tail_epochs"] == "8"
    assert form["lr_scheduler_min_lr_ratio"] == "0.05"
    assert "constantcosine" not in form.get("extra_flags", "")


def test_load_auto_preprocess_orchestration_keys():
    # GUI auto-preprocess knobs are not train.py args — they must map to their
    # dedicated form fields, never leak into extra_flags as bogus --flags.
    form = load_toml_to_form(
        "auto_preprocess = true\n"
        "multiscale = true\n"
        "drop_lowres = false\n"
        "caption_shuffle_variants = 6\n"
    )
    assert form["auto_preprocess"] is True and form["multiscale"] is True
    assert form["drop_lowres"] is False
    assert form["caption_shuffle_variants"] == "6"
    assert "auto_preprocess" not in form.get("extra_flags", "")


def test_load_resume_and_caption_variant_fields():
    form = load_toml_to_form(
        'resume = "output/ckpt/x-state"\n'
        "cache_text_encoder_outputs_to_disk = true\n"  # LETS rename → use_text_cache
        "use_shuffled_caption_variants = true\n"
    )
    assert form["resume"] == "output/ckpt/x-state"
    assert form["use_text_cache"] is True
    assert form["use_shuffled_caption_variants"] is True


_MULTI_SUBSET = """
gradient_checkpointing_resolutions = [512, 1024]
[[datasets]]
batch_size = 2
  [[datasets.subsets]]
  image_dir = "data/main"
  num_repeats = 5
  [[datasets.subsets]]
  image_dir = "data/reg"
  flip_aug = true
  tiers = [512, 1024]
[[datasets]]
batch_size = 4
  [[datasets.subsets]]
  image_dir = "data/third"
"""


def test_load_multi_subset_fills_blocks():
    form = load_toml_to_form(_MULTI_SUBSET)
    # subset #1 → primary ds_* block (batch_size from its owning [[datasets]] block)
    assert form["ds_image_dir"] == "data/main"
    assert form["ds_num_repeats"] == "5"
    assert form["ds_batch_size"] == "2"
    # subset #2 → ds2_* block
    assert form["ds2_image_dir"] == "data/reg"
    assert form["ds2_flip_aug"] is True
    assert form["ds2_tiers"] == "512,1024"
    assert form["ds2_batch_size"] == "2"  # first block's batch_size
    # per-subset gradient_checkpointing restored by tier match (512,1024 ⊂ edges)
    assert form["ds2_gradient_checkpointing"] is True
    # subset #3 → ds3_* block (second [[datasets]] block, batch_size 4)
    assert form["ds3_image_dir"] == "data/third"
    assert form["ds3_batch_size"] == "4"
    # gradient_checkpointing_resolutions is consumed (NOT echoed into extra_flags)
    assert "gradient_checkpointing_resolutions" not in (form.get("extra_flags") or "")


def test_load_drops_sd_era_crash_keys():
    """kohya / LoRA_Easy / SD-era keys with no anima flag must be DROPPED on load,
    never folded into extra_flags — train.py uses a strict argparse, so a leaked
    --shuffle_caption / --sdxl / … would abort the run at Start. Regression guard for
    the config_io._DROP ↔ backend._IMPORT_DROP drift the audit caught."""
    form = load_toml_to_form(
        "shuffle_caption = true\n"
        "skip_image_resolution = false\n"
        "clip_skip = 2\n"
        "sdxl = true\n"
        "v2 = false\n"
        "v_parameterization = false\n"
        'name = "my_lora"\n'
    )
    ef = form.get("extra_flags", "")
    for k in (
        "shuffle_caption",
        "skip_image_resolution",
        "clip_skip",
        "sdxl",
        "v2",
        "v_parameterization",
        "name",
    ):
        assert k not in ef, f"{k} leaked into extra_flags → would crash train.py"


def test_load_preserves_valid_anima_flags():
    """Anima-VALID flags must survive load. prior_loss_weight (float) rides emit();
    no_half_vae is a store_true → emit `--no_half_vae` only when true (a false value
    must NOT become `--no-no_half_vae`, which argparse would reject)."""
    ef = load_toml_to_form("prior_loss_weight = 5.0\n").get("extra_flags", "")
    assert "--prior_loss_weight 5.0" in ef
    assert "--no_half_vae" in load_toml_to_form("no_half_vae = true\n").get(
        "extra_flags", ""
    )
    assert "no_half_vae" not in load_toml_to_form("no_half_vae = false\n").get(
        "extra_flags", ""
    )
    # lowram is owned by _BOOL_FIELDS (its dead _DROP entry was removed).
    assert load_toml_to_form("lowram = true\n")["lowram"] is True


_SAME_DIR_MULTISCALE = """
[[datasets]]
resolution = [1536, 1536]
skip_image_resolution = [1024, 1024]
batch_size = 1
  [[datasets.subsets]]
  image_dir = "C:/x/imgs"
  num_repeats = 10
  keep_tokens = 5
[[datasets]]
resolution = [1024, 1024]
batch_size = 1
  [[datasets.subsets]]
  image_dir = "C:/x/imgs"
  num_repeats = 10
  keep_tokens = 5
[[datasets]]
resolution = [512, 512]
batch_size = 1
  [[datasets.subsets]]
  image_dir = "C:/x/imgs"
  num_repeats = 10
  keep_tokens = 5
"""


def test_load_dedups_same_dir_multiscale_to_one_subset():
    """A LoRA_Easy "same folder at N resolutions" dataset_config collapses to ONE
    subset + the union of tiers (anima multiscale). skip_image_resolution is dropped."""
    form = load_toml_to_form(_SAME_DIR_MULTISCALE)
    assert form["ds_image_dir"] == "C:/x/imgs"
    assert form["ds_num_repeats"] == "10" and form["ds_keep_tokens"] == "5"
    assert "ds2_image_dir" not in form  # deduped — only one distinct folder
    assert sorted(form["target_res"]) == ["1024", "1536", "512"]  # union of tiers
    assert "skip_image_resolution" not in form.get("extra_flags", "")


def test_known_dests_pass_schema_args_to_fields_not_extra_flags():
    """Schema/advanced args (no dedicated curated field) are folded into extra_flags
    by default, but when the caller passes ``known_dests`` (the native GUI's full
    field set) they pass through to their own form key so the field/dropdown
    populates on load. Regression for "config load: dropdowns/values don't apply"."""
    toml_text = (
        'method = "lora"\n'
        'dynamo_backend = "inductor"\n'
        "activation_memory_budget = 0.95\n"
        "cache_info = true\n"
    )
    # default: no known_dests → these land in extra_flags (legacy behavior)
    plain = load_toml_to_form(toml_text)
    assert "dynamo_backend" not in plain
    assert "dynamo_backend" in (plain.get("extra_flags") or "")
    # with known_dests → they reach their fields, NOT extra_flags
    known = load_toml_to_form(
        toml_text,
        known_dests={"dynamo_backend", "activation_memory_budget", "cache_info"},
    )
    assert known.get("dynamo_backend") == "inductor"
    assert known.get("activation_memory_budget") == 0.95
    assert known.get("cache_info") is True
    assert "dynamo_backend" not in (known.get("extra_flags") or "")


def test_subset_cards_round_trip():
    """The native GUI's dynamic subset cards (form['subsets']) survive save→load:
    image_dir + caption_extension dropdown + flip_aug/is_val toggles + numeric values
    + per-subset tiers + reconstructed gradient_checkpointing. Regression for "saved
    config drops subsets / their dropdowns + ON/OFF toggles don't restore"."""
    form = {
        "method": "lora",
        "preset": "default",
        "subsets": [
            {
                "image_dir": "image_dataset/cat",
                "num_repeats": "4",
                "keep_tokens": "1",
                "caption_extension": ".txt",
                "caption_dropout_rate": "0.1",
                "flip_aug": True,
                "random_crop": False,
                "is_val": False,
                "gradient_checkpointing": True,
                "tiers": "1024,768",
                "batch_size": "2",
            },
            {"image_dir": "image_dataset/val", "num_repeats": "1", "is_val": True},
        ],
    }
    back = load_toml_to_form(save_form_to_toml(form))
    subs = back.get("subsets")
    assert isinstance(subs, list) and len(subs) == 2, subs
    a, b = subs
    assert a["image_dir"] == "image_dataset/cat"
    assert a["caption_extension"] == ".txt"  # dropdown
    assert a["num_repeats"] == "4" and a["keep_tokens"] == "1"  # values
    assert a["flip_aug"] is True and "random_crop" not in a  # ON kept, OFF dropped
    assert a["tiers"] == "1024,768"
    assert a["gradient_checkpointing"] is True  # reconstructed from gc_resolutions
    assert b["image_dir"] == "image_dataset/val" and b["is_val"] is True
    # logging_dir is derived on save → must NOT leak into extra_flags on load
    assert "--logging_dir" not in (back.get("extra_flags") or "")


def test_output_dir_does_not_nest_on_round_trip():
    """save emits the COMPUTED output_dir (<base>/<name>); persisting that verbatim
    would re-nest a level deeper every save→load cycle. Regression: it stays put."""
    form = {"method": "lora", "output_name": "mylora", "output_dir": "output"}
    for _ in range(3):
        text = save_form_to_toml(form)
        form = load_toml_to_form(text)
        assert form.get("output_dir") == "output"
        assert "--logging_dir" not in (form.get("extra_flags") or "")


def test_native_subsets_dedup_same_folder():
    """A kohya multiscale config lists the same folder under N resolution blocks —
    collapse to ONE native subset card (mirrors the ds_* dedup)."""
    text = (
        "[[datasets]]\nresolution = 768\n  [[datasets.subsets]]\n"
        '  image_dir = "C:/d/char"\n  num_repeats = 3\n'
        "[[datasets]]\nresolution = 1024\n  [[datasets.subsets]]\n"
        '  image_dir = "C:/d/char"\n  num_repeats = 3\n'
    )
    subs = load_toml_to_form(text).get("subsets")
    assert isinstance(subs, list) and len(subs) == 1, subs
    assert subs[0]["image_dir"] == "C:/d/char"


if __name__ == "__main__":  # allow `python tests/test_config_io.py`
    test_subset_cards_round_trip()
    test_output_dir_does_not_nest_on_round_trip()
    test_native_subsets_dedup_same_folder()
    test_known_dests_pass_schema_args_to_fields_not_extra_flags()
    test_load_maps_dedicated_fields()
    test_load_renames_to_dedicated_fields()
    test_load_drops_and_extra_routing()
    test_save_emits_runnable_toml_and_round_trips()
    test_load_harvests_kohya_dataset_block()
    test_load_harvests_anima_dataset_tiers()
    test_load_without_dataset_leaves_ds_fields_unset()
    test_load_multi_subset_fills_blocks()
    test_load_constantcosine_fields()
    test_load_auto_preprocess_orchestration_keys()
    test_load_resume_and_caption_variant_fields()
    test_load_drops_sd_era_crash_keys()
    test_load_preserves_valid_anima_flags()
    test_load_dedups_same_dir_multiscale_to_one_subset()
    print("all config_io round-trip tests passed")
