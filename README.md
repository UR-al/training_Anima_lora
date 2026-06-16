# Anima LoRA — merged trainer

A fast **Anima** LoRA trainer with a browser control panel, a ~89-optimizer zoo, custom LR schedulers, and a live training dashboard — three projects merged into one.

## What it is

| From | What | Vendored as | License |
|---|---|---|---|
| **[anima_lora](https://github.com/sorryhyun/anima_lora)** | the fast base: `torch.compile` LoRA training of the Anima DiT (constant-token bucketing + native-flatten compile, flash-attn, block-swap, fully-cached dataloader) | the repo itself | MIT |
| **[LoRA_Easy_Training_Scripts](https://github.com/67372a/LoRA_Easy_Training_Scripts)** | the broad **optimizer + scheduler** suite (~89 optimizers, CAWR/RAWR schedulers) | `custom_scheduler/LoraEasyCustomOptimizer/` | GPL-3.0 |
| **[AnimaLoraToolkit](https://github.com/Moeblack/AnimaLoraToolkit)** | the live **web monitor** (loss / LR / sample dashboard) | `library/monitoring/` | GPL-3.0 |

> The combined, distributed work is **GPL-3.0** — see [License & attribution](#license--attribution). The **model weights** are governed by a separate **non-commercial** license; that section is required reading before you publish anything you train.

## Quickstart (Windows)

```powershell
git clone https://github.com/UR-al/training_Anima_lora
cd training_Anima_lora
install_uv.bat                 REM via uv (recommended). install_pip.bat = pip alternative (needs Python 3.13)
run_gui.bat                    REM opens the Gradio control panel in your browser
```

Then, in the GUI's **Model files** panel (or via `make download-models`), fetch the Anima DiT + Qwen3 text encoder + VAE — they are **not** shipped in this repo. Point the dataset at a folder of images with `.txt` caption sidecars, toggle **Auto-preprocess at train start**, and hit **Start**. `update.bat` later pulls + re-syncs.

## Gradio control panel — `run_gui.bat`

`run_gui.bat` (or `python tasks.py gradio-gui`) opens the **Gradio** control panel — a kohya_ss-style layout wired to **this repo's** `train.py`. Opt-in `gradio` extra: `uv sync --extra gradio` (the launchers do this for you).

- **Tabs:** Model / LoRA / Dataset / Samples / Training / Config / Utils. Choose **method / preset / optimizer (~89) / scheduler** from dropdowns; set rank, LR, epochs, seed, optimizer & scheduler args.
- **Dataset:** build subsets by hand (single- or multi-subset, per-subset repeats / keep-tokens / tiers / batch-size), **or** point at a `dataset_config` TOML and click **Load → fill subsets** to import a LoRA_Easy / anima dataset (same-folder multi-resolution blocks collapse to one subset + the union of tiers).
- **Config load/save:** round-trips anima configs and is compatible with **LoRA_Easy / kohya** configs — key renames are applied, unmapped keys fall to an *Extra CLI flags* box, and SD-era keys with no Anima equivalent are dropped (so a foreign config loads without crashing `train.py`).
- **Guidance:** per-field help tooltips; dependency **greying with reasons** (a disabled field shows *why*); optimizer-arg help on the left, scheduler-arg help on the right.
- **Run:** **preview the command**, then **Start** / **Stop**. Training runs as a **direct `train.py` subprocess**; stdout/stderr is captured to `output/logs/` and tailed live in the panel. There is no job queue daemon — a saved-run **Queue** launches runs sequentially.
- **Extras:** Auto-preprocess at train start, a **Utils** tab (SAM3 / MIT masking + auto-batch search), and a one-click link to the **live monitor** (loss / LR / sample dashboard; rehydrates the curve on `--resume`).

> **Prefer a desktop app?** A native **PySide6** control panel ships too — `run_gui_native.bat` (or `python tasks.py native-gui`; opt-in `gui` extra: `uv sync --extra gui`) — as an alternative to the browser panel. It drives the same `train.py`.

## CLI

```powershell
python tasks.py lora --optimizer_type CAME --monitor                              REM named optimizer + dashboard
python tasks.py lora --method lora --preset low_vram --dataset_config my.toml --network_dim 32
```

`--optimizer_type <name>` takes a friendly name (`CAME`, `ADOPT`, `Prodigy`, `ProdigyPlusScheduleFree`, …) or any `pkg.module.Class`. `--lr_scheduler_type <dotted path>` selects a custom scheduler. `--monitor` starts the dashboard. See `CLAUDE.md` for the full reference.

## LoKr / LoHa & the full LyCORIS zoo

The whole **LyCORIS** family (LoKr, LoHa, DyLoRA, GLoRA, (IA)³, Diag-OFT, BOFT, Full) trains on the Anima DiT **with the `torch.compile` speed core intact**. In the GUI just set **Adapter → `networks.lycoris_anima`** and pick an **algo** + **preset** (`anima-attn-mlp` = attention+MLP, 197 modules; `anima-full` = +adaln/embeds, 314). From the CLI:

```powershell
REM LoKr (factor decomposition; no grad-checkpointing needed)
python tasks.py lora --method lycoris --network_args algo=lokr preset=configs/lycoris_presets/anima_attn_mlp.toml factor=4 full_matrix=True

REM LoHa (materializes full ΔW per module — add --gradient_checkpointing + a small rank on <=16 GB)
python tasks.py lora --method lycoris --network_dim 16 --network_alpha 8 --gradient_checkpointing --network_args algo=loha preset=configs/lycoris_presets/anima_attn_mlp.toml
```

`networks.lycoris_anima` bridges stock `lycoris-lora` to the Anima DiT (the upstream presets target diffusers class names the Anima blocks don't use). See `configs/lycoris_presets/` and `configs/methods/lycoris.toml`.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 (8 GB) | 16 GB+ |
| System RAM | 16 GB | 32 GB+ |
| Disk | 60 GB | 200 GB+ |

Python 3.13 + PyTorch 2.12 (cu132) are installed for you. `torch.compile` needs the CUDA 13.2 toolkit (nvcc).

## Install (Windows)

```powershell
git clone https://github.com/UR-al/training_Anima_lora
cd training_Anima_lora
install_uv.bat     REM via uv (recommended).  install_pip.bat = pip alternative (needs Python 3.13)
```

For a manual / CI pip install, `requirements.txt` mirrors the dependencies (cu132 torch index + `--pre` are baked in): `pip install -r requirements.txt && pip install -e . --no-deps`. (`pyproject.toml` + `uv.lock` remain the uv source of truth — keep `requirements.txt` in sync when adding a library.)

Fetch the model weights (Anima DiT + Qwen3 text encoder + VAE) into `models/` — they are **not** shipped in this repo (gitignored) — or point the GUI's **Model files** panel at existing forge-neo / ComfyUI files.

## License & attribution

This repository bundles code from three projects, so **two independent license layers apply** — the **code** and the **model weights**. Read both.

### Code — GPL-3.0

The combined work distributed in this repository is licensed under **GPL-3.0** (see [`LICENSE`](LICENSE)). The copyleft is load-bearing: the repo **vendors GPL-3.0 code**, so the distribution as a whole must be GPL-3.0.

| Component | Upstream license | Lives in | Note |
|---|---|---|---|
| **anima_lora** base engine | **MIT** | the repo itself | the original toolkit code (© 2026 Seunghyun Ji); MIT text kept as [`LICENSE-MIT`](LICENSE-MIT) |
| **kohya-ss/sd-scripts** derivation | **Apache-2.0** | `library/`, `networks/` | this repo was originally adapted from sd-scripts; those portions stay Apache-2.0 ([`LICENSE-APACHE`](LICENSE-APACHE)); modifications stated in [`NOTICE`](NOTICE) |
| **LoRA_Easy_Training_Scripts** optimizer/scheduler zoo | **GPL-3.0** | `custom_scheduler/LoraEasyCustomOptimizer/` | vendored; individual optimizer files keep their original author headers (Apache-2.0 / MIT / BSD-3-Clause) |
| **AnimaLoraToolkit** web monitor | **GPL-3.0** | `library/monitoring/` | derives from ComfyUI (GPL-3.0) |

The two GPL-3.0 components above (the optimizer zoo and the monitor) are why the combined work is GPL-3.0. Under GPL-3.0 you may use, modify, and redistribute the code — including commercially — provided you keep derivatives under GPL-3.0 and offer source for what you distribute. The permissive MIT/Apache portions retain their original licenses where separable; combining them here does not strip those notices (`LICENSE-MIT` / `LICENSE-APACHE` / `NOTICE`).

### Model weights — separate, **non-commercial**

The Anima base weights are **not** covered by the code license. They are published by CircleStone Labs LLC under the **CircleStone Labs Non-Commercial License v1.0 (NCL)**:

- The Anima / CircleStone base weights stay under the NCL — obtain them from their original source and comply with it.
- **LoRA adapters, fine-tunes, and merges trained from the Anima weights are "Derivatives" under the NCL and inherit its non-commercial terms** — regardless of the GPL-3.0 on this code.
- Train on a *different* base model you hold commercial rights to, and the NCL does not attach; only the code license applies to those adapters.

The full NCL text ships with the weights (not redistributed here, to avoid staleness). See <https://huggingface.co/CircleStoneLab> or the model card where you obtained the weights. Full detail in [`NOTICE`](NOTICE).
