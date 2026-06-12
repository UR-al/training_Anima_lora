# -*- coding: utf-8 -*-
"""
Vendored from LoRA_Easy_Training_Scripts (67372a) for the merged Anima trainer.

This is an IMPORT-GUARDED rewrite of the upstream __init__.py. Upstream imported
every optimizer eagerly, so a single missing optional dependency
(bitsandbytes / torchao / adv_optm / prodigyplus) broke import of the WHOLE package
— and with it every other optimizer. Here each optimizer is imported defensively:
unavailable ones are skipped and recorded, the package always imports, and the
`OPTIMIZERS` registry reflects only what is actually loadable on the current stack.

Public surface is unchanged for available optimizers:
  - `OPTIMIZER_LIST`  : list[type] of loadable optimizer classes
  - `OPTIMIZERS`      : {classname.lower(): class} (anima_lora's resolver / the
                        friendly-name rewrite key off this)
  - each available class is also bound as a module attribute (e.g. `CAME`)
Extra helpers added for the merge:
  - `available_optimizers()` -> sorted list of registry keys
  - `skipped_optimizers()`   -> {name: reason} for the unavailable ones
"""
import importlib
import logging
from typing import Dict, List

_log = logging.getLogger(__name__)

try:
    from LoraEasyCustomOptimizer.utils import OPTIMIZER  # type alias used in annotations
except Exception:  # pragma: no cover - utils should always import, but stay defensive
    OPTIMIZER = "type"


def _install_compat_shims() -> None:
    """Make the quantized (torchao / bitsandbytes) optimizers importable on the
    merged tool's bleeding-edge stack (torch 2.12 / cu132).

    1. torchao: the vendored ``low_bit_optim`` imports TORCH_VERSION_AT_LEAST_2_4/_5
       from ``torchao.utils``, which newer torchao removed. We're on torch >= 2.5,
       so re-provide them (and _2_6) as booleans when absent.
    2. bitsandbytes: ships CUDA binaries only up to cuda130 (no cu132). On CUDA
       13.x, default BNB_CUDA_VERSION=130 so it loads the forward-compatible
       cuda130 binary instead of erroring on a missing libbitsandbytes_cuda132.dll.
    Both are best-effort and fully guarded — failure leaves the stack as-is.
    """
    import os
    import re

    try:
        import torch

        cuda_ver = getattr(torch.version, "cuda", None)  # e.g. "13.2"
        if cuda_ver and cuda_ver.split(".")[0] == "13":
            os.environ.setdefault("BNB_CUDA_VERSION", "130")

        m = re.match(r"(\d+)\.(\d+)", torch.__version__ or "")
        tv = (int(m.group(1)), int(m.group(2))) if m else (2, 12)
        import torchao.utils as _tao

        for _name, _need in (
            ("TORCH_VERSION_AT_LEAST_2_4", (2, 4)),
            ("TORCH_VERSION_AT_LEAST_2_5", (2, 5)),
            ("TORCH_VERSION_AT_LEAST_2_6", (2, 6)),
        ):
            if not hasattr(_tao, _name):
                setattr(_tao, _name, tv >= _need)
    except Exception:
        pass


_install_compat_shims()

# (module, [(import_name, export_name), ...]) — mirrors upstream __init__ exactly,
# including the adv_optm alias (Simplified_AdEMAMix -> Simplified_AdEMAMix_adv).
_SPECS = [
    ("LoraEasyCustomOptimizer.adabelief", [("AdaBelief", "AdaBelief")]),
    ("LoraEasyCustomOptimizer.adagc", [("AdaGC", "AdaGC")]),
    ("LoraEasyCustomOptimizer.adammini", [("AdamMini", "AdamMini")]),
    ("LoraEasyCustomOptimizer.adan", [("Adan", "Adan")]),
    ("LoraEasyCustomOptimizer.ademamix", [("AdEMAMix", "AdEMAMix"), ("SimplifiedAdEMAMix", "SimplifiedAdEMAMix"), ("SimplifiedAdEMAMixExM", "SimplifiedAdEMAMixExM")]),
    ("LoraEasyCustomOptimizer.adopt", [("ADOPT", "ADOPT")]),
    ("LoraEasyCustomOptimizer.came", [("CAME", "CAME")]),
    ("LoraEasyCustomOptimizer.compass", [("Compass", "Compass"), ("Compass8BitBNB", "Compass8BitBNB"), ("CompassPlus", "CompassPlus"), ("CompassADOPT", "CompassADOPT"), ("CompassADOPTMARS", "CompassADOPTMARS"), ("CompassAO", "CompassAO")]),
    ("LoraEasyCustomOptimizer.farmscrop", [("FARMSCrop", "FARMSCrop"), ("FARMSCropV2", "FARMSCropV2")]),
    ("LoraEasyCustomOptimizer.fcompass", [("FCompass", "FCompass"), ("FCompassPlus", "FCompassPlus"), ("FCompassADOPT", "FCompassADOPT"), ("FCompassADOPTMARS", "FCompassADOPTMARS")]),
    ("LoraEasyCustomOptimizer.fishmonger", [("FishMonger", "FishMonger"), ("FishMonger8BitBNB", "FishMonger8BitBNB")]),
    ("LoraEasyCustomOptimizer.fmarscrop", [("FMARSCrop", "FMARSCrop"), ("FMARSCropV2", "FMARSCropV2"), ("FMARSCropV2ExMachina", "FMARSCropV2ExMachina"), ("FMARSCropV3", "FMARSCropV3"), ("FMARSCropV3ExMachina", "FMARSCropV3ExMachina")]),
    ("LoraEasyCustomOptimizer.galore", [("GaLore", "GaLore")]),
    ("LoraEasyCustomOptimizer.gooddog", [("GOODDOG", "GOODDOG")]),
    ("LoraEasyCustomOptimizer.grokfast", [("GrokFastAdamW", "GrokFastAdamW")]),
    ("LoraEasyCustomOptimizer.laprop", [("LaProp", "LaProp")]),
    ("LoraEasyCustomOptimizer.lpfadamw", [("LPFAdamW", "LPFAdamW")]),
    ("LoraEasyCustomOptimizer.ranger21", [("Ranger21", "Ranger21")]),
    ("LoraEasyCustomOptimizer.spam", [("StableSPAM", "StableSPAM")]),
    ("LoraEasyCustomOptimizer.rmsprop", [("RMSProp", "RMSProp"), ("RMSPropADOPT", "RMSPropADOPT"), ("RMSPropADOPTMARS", "RMSPropADOPTMARS")]),
    ("LoraEasyCustomOptimizer.schedulefree", [("ScheduleFreeWrapper", "ScheduleFreeWrapper"), ("ADOPTScheduleFree", "ADOPTScheduleFree"), ("ADOPTEMAMixScheduleFree", "ADOPTEMAMixScheduleFree"), ("ADOPTNesterovScheduleFree", "ADOPTNesterovScheduleFree"), ("FADOPTScheduleFree", "FADOPTScheduleFree"), ("ADOPTMARSScheduleFree", "ADOPTMARSScheduleFree"), ("FADOPTMARSScheduleFree", "FADOPTMARSScheduleFree"), ("ADOPTAOScheduleFree", "ADOPTAOScheduleFree")]),
    ("LoraEasyCustomOptimizer.clybius_experiments", [("MomentusCaution", "MomentusCaution"), ("REMASTER", "REMASTER")]),
    ("LoraEasyCustomOptimizer.scion", [("SCION", "SCION")]),
    ("LoraEasyCustomOptimizer.sgd", [("SGDSaI", "SGDSaI")]),
    ("LoraEasyCustomOptimizer.shampoo", [("ScalableShampoo", "ScalableShampoo")]),
    ("LoraEasyCustomOptimizer.adam", [("AdamW8bitAO", "AdamW8bitAO"), ("AdamW4bitAO", "AdamW4bitAO"), ("AdamWfp8AO", "AdamWfp8AO"), ("AdamW8bitKahan", "AdamW8bitKahan")]),
    ("prodigyplus.prodigy_plus_schedulefree", [("ProdigyPlusScheduleFree", "ProdigyPlusScheduleFree")]),
    ("LoraEasyCustomOptimizer.scorn", [("SCORN", "SCORN")]),
    ("LoraEasyCustomOptimizer.scornmachina", [("SCORNMachina", "SCORNMachina")]),
    ("LoraEasyCustomOptimizer.mythical", [("Mythical", "Mythical")]),
    ("LoraEasyCustomOptimizer.oagopt", [("OAGOpt", "OAGOpt")]),
    ("LoraEasyCustomOptimizer.ocgopt", [("OCGOpt", "OCGOpt")]),
    ("LoraEasyCustomOptimizer.ocgoptv2", [("OCGOptV2", "OCGOptV2")]),
    ("LoraEasyCustomOptimizer.glyph", [("Glyph", "Glyph")]),
    ("LoraEasyCustomOptimizer.racs", [("RACS", "RACS")]),
    ("LoraEasyCustomOptimizer.alice", [("Alice", "Alice")]),
    ("LoraEasyCustomOptimizer.fira", [("Fira", "Fira")]),
    ("LoraEasyCustomOptimizer.vsgd", [("VSGD", "VSGD")]),
    ("LoraEasyCustomOptimizer.cstableadamw", [("CStableAdamW", "CStableAdamW")]),
    ("LoraEasyCustomOptimizer.dehaze", [("Dehaze", "Dehaze")]),
    ("LoraEasyCustomOptimizer.talon", [("TALON", "TALON")]),
    ("LoraEasyCustomOptimizer.fftdescent", [("FFTDescent", "FFTDescent")]),
    ("LoraEasyCustomOptimizer.scgopt", [("SCGOpt", "SCGOpt")]),
    ("LoraEasyCustomOptimizer.singstate", [("SingState", "SingState")]),
    ("LoraEasyCustomOptimizer.snoo_asgd", [("SNOO_ASGD", "SNOO_ASGD")]),
    ("adv_optm.optim", [("AdamW_adv", "AdamW_adv"), ("Adopt_adv", "Adopt_adv"), ("Simplified_AdEMAMix", "Simplified_AdEMAMix_adv"), ("Lion_adv", "Lion_adv")]),
    ("LoraEasyCustomOptimizer.abmog", [("ABMOG", "ABMOG")]),
    ("LoraEasyCustomOptimizer.bcos", [("BCOS", "BCOS")]),
    ("LoraEasyCustomOptimizer.projective_adam", [("ProjectiveAdam", "ProjectiveAdam")]),
    ("LoraEasyCustomOptimizer.wiwiopt", [("WiwiOpt", "WiwiOpt")]),
    ("LoraEasyCustomOptimizer.cascade", [("CASCADE", "CASCADE")]),
    ("LoraEasyCustomOptimizer.radam_schedulefree", [("RAdamScheduleFree", "RAdamScheduleFree")]),
    ("LoraEasyCustomOptimizer.nor_muon_schedulefree", [("NorMuonScheduleFree", "NorMuonScheduleFree")]),
    ("LoraEasyCustomOptimizer.adamw_schedulefree_plus", [("AdamWScheduleFreePlus", "AdamWScheduleFreePlus")]),
    ("LoraEasyCustomOptimizer.amuse", [("AMUSE", "AMUSE")]),
    ("LoraEasyCustomOptimizer.soda", [("SODA", "SODA")]),
    ("LoraEasyCustomOptimizer.moda", [("MODA", "MODA")]),
    ("LoraEasyCustomOptimizer.soda_wrapper", [("SODAWrapper", "SODAWrapper")]),
]

OPTIMIZER_LIST: List[type] = []
OPTIMIZERS: Dict[str, type] = {}
_SKIPPED: Dict[str, str] = {}


def _load() -> None:
    for module_path, names in _SPECS:
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # missing optional dep, or module-load failure
            reason = f"{type(exc).__name__}: {exc}"
            for _src, dst in names:
                _SKIPPED[dst] = reason
            _log.debug("LoraEasyCustomOptimizer: module %s unavailable (%s)", module_path, reason)
            continue
        for src_name, export_name in names:
            cls = getattr(module, src_name, None)
            if cls is None:
                _SKIPPED[export_name] = f"name {src_name!r} not found in {module_path}"
                continue
            globals()[export_name] = cls
            OPTIMIZER_LIST.append(cls)
            OPTIMIZERS[cls.__name__.lower()] = cls


def _missing_dep_names() -> List[str]:
    """Best-effort list of the optional packages whose absence skipped optimizers."""
    deps = set()
    for reason in _SKIPPED.values():
        marker = "No module named "
        if marker in reason:
            deps.add(reason.split(marker, 1)[1].strip().strip("'\""))
    return sorted(deps)


_load()

if _SKIPPED:
    _log.info(
        "LoraEasyCustomOptimizer: %d optimizer(s) available, %d skipped. Missing optional deps: %s",
        len(OPTIMIZER_LIST),
        len(_SKIPPED),
        ", ".join(_missing_dep_names()) or "(see skipped_optimizers())",
    )


def available_optimizers() -> List[str]:
    """Sorted registry keys (classname.lower()) that are actually loadable here."""
    return sorted(OPTIMIZERS.keys())


def skipped_optimizers() -> Dict[str, str]:
    """Map of export-name -> reason for optimizers unavailable on this stack."""
    return dict(_SKIPPED)
