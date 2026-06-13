# Anima LoRA — merged trainer

A fast **Anima** LoRA trainer with a browser control panel, a ~89-optimizer zoo, custom LR schedulers, and a live training dashboard — three projects merged into one.

## What it is

| From | What | Vendored as |
|---|---|---|
| **[anima_lora](https://github.com/sorryhyun/anima_lora)** (MIT) | the fast base: `torch.compile` LoRA training of the Anima DiT (constant-token bucketing + native-flatten compile, flash-attn, block-swap, fully-cached dataloader) | the repo itself |
| **LoRA_Easy_Training_Scripts** | the broad **optimizer + scheduler** suite (~89 optimizers, CAWR/RAWR schedulers) | `LoraEasyCustomOptimizer/` |
| **AnimaLoraToolkit** (GPL-3.0) | the live **web monitor** (loss / LR / sample dashboard) | `library/monitoring/` |

## Web control panel — `run_gui.bat`

Run `run_gui.bat` (or `python tasks.py webgui`) to open a browser control panel — **no Qt, pure stdlib**:

- Pick **method / preset / optimizer (~89) / scheduler** from dropdowns; set rank, LR, epochs, dataset, seed, optimizer/scheduler args.
- **Preview the command**, then **Start** / **Stop**.
- **Runs through the training daemon by default**, so training survives closing the page (and queues + captures logs).
- One-click link to the **live monitor** (loss/LR/sample dashboard, resumes the curve on `--resume`).

## CLI

```powershell
python tasks.py lora --optimizer_type CAME --monitor     REM train with a named optimizer + dashboard
python tasks.py lora --method lora --preset low_vram --dataset_config my.toml --network_dim 32
```

`--optimizer_type <name>` takes a friendly name (`CAME`, `ADOPT`, `Prodigy`, `ProdigyPlusScheduleFree`, …) or any `pkg.module.Class`. `--lr_scheduler_type <dotted path>` for custom schedulers. `--monitor` for the dashboard. See `CLAUDE.md` for the full reference.

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
install.bat        REM via uv (recommended).  install_pip.bat = pip alternative (needs Python 3.13)
```

Then fetch the model weights (Anima DiT + Qwen3 text encoder + VAE) into `models/` — they are **not** shipped in this repo (gitignored). `update.bat` later pulls + re-syncs.

## License & attribution

This is a **derivative work** of the three projects above:

- **anima_lora** — **MIT** (the base engine; its notice is preserved as `LICENSE-MIT`).
- **LoRA_Easy_Training_Scripts** — its optimizer/scheduler package is vendored under `LoraEasyCustomOptimizer/` (retains upstream authorship; see that tree's headers).
- **AnimaLoraToolkit** — its web monitor is vendored under `library/monitoring/`. **AnimaLoraToolkit is GPL-3.0** (it derives from ComfyUI's GPL-3.0 model code). The monitor files carry no separate license, so they are covered by that GPL-3.0.

**This combined work is licensed under GPL-3.0** (`LICENSE`) because it includes the GPL-3.0 monitor; the permissive MIT/Apache notices of the included parts are kept as `LICENSE-MIT` / `LICENSE-APACHE` and in `NOTICE`. Model weights (Anima / Qwen / VAE) have their own terms — check each model card.
