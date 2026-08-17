import math
import threading

import pytest
import torch

from library.runtime.dynamo import pin_inductor_flag
from networks import NETWORK_KWARGS
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_modules import LoRAModule
from networks.lora_save import _relayout_adaln_to_comfy
from networks.lora_utils import (
    has_comfy_adaln_keys,
    relayout_adaln_comfy_to_runtime,
)


def _cfg(kwargs: dict[str, str]) -> LoRANetworkCfg:
    return LoRANetworkCfg.from_kwargs(
        kwargs,
        network_dim=32,
        network_alpha=32.0,
        neuron_dropout=None,
        module_class=LoRAModule,
    )


def test_train_adaln_rescues_targets_and_derives_scale() -> None:
    cfg = _cfg({"train_adaln": "true", "adaln_rank": "16"})

    assert ".*adaln_up_.*" in cfg.include_patterns
    assert cfg.reg_dims[".*adaln_up_.*"] == 16
    assert cfg.reg_alphas[".*adaln_up_.*"] == pytest.approx(32.0 / math.sqrt(2.0))


def test_adaln_top_level_config_keys_are_forwarded() -> None:
    assert {
        "train_adaln",
        "adaln_rank",
        "adaln_alpha",
        "network_reg_alphas",
    } <= NETWORK_KWARGS


@pytest.mark.parametrize("key", ["adaln_rank", "adaln_alpha"])
def test_adaln_overrides_require_training_enabled(key: str) -> None:
    with pytest.raises(ValueError, match="train_adaln"):
        _cfg({key: "16"})


def test_adaln_checkpoint_relayout_round_trips() -> None:
    runtime_key = "lora_unet_blocks_0_adaln_up_mlp.lora_down.weight"
    comfy_key = "lora_unet_blocks_0_adaln_modulation_mlp_2.lora_down.weight"
    state = {runtime_key: torch.randn(4, 8)}

    metadata = _relayout_adaln_to_comfy(state, None)

    assert runtime_key not in state
    assert comfy_key in state
    assert metadata == {"ss_adaln_layout": "comfy"}
    assert has_comfy_adaln_keys(state)
    restored = relayout_adaln_comfy_to_runtime(state)
    assert runtime_key in restored


def test_pin_inductor_flag_holds_in_fresh_context() -> None:
    import torch._inductor.config as config
    from torch.utils._config_module import _UNSET_SENTINEL

    key = "triton.mix_order_reduction"
    entry = config._config[key]
    original_default = entry.default
    original_override = entry.user_override.get(_UNSET_SENTINEL)
    observed: dict[str, object] = {}

    try:
        entry.default = True
        entry.user_override.set(_UNSET_SENTINEL)
        pin_inductor_flag(key, False)

        thread = threading.Thread(
            target=lambda: observed.setdefault(
                "value", config.triton.mix_order_reduction
            )
        )
        thread.start()
        thread.join()
        assert observed["value"] is False
    finally:
        entry.default = original_default
        entry.user_override.set(original_override)
