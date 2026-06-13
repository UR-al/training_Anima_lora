# Anima LoRA — merged trainer

A fast **Anima** LoRA trainer with a browser control panel, a ~89-optimizer zoo, custom LR schedulers, and a live training dashboard — three projects merged into one.

## What it is

| From | What | Vendored as |
|---|---|---|
| **[anima_lora](https://github.com/sorryhyun/anima_lora)** (Apache-2.0) | the fast base: `torch.compile` LoRA training of the Anima DiT (constant-token bucketing + native-flatten compile, flash-attn, block-swap, fully-cached dataloader) | the repo itself |
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

- **anima_lora** — Apache-2.0 (the base engine).
- **LoRA_Easy_Training_Scripts** — its optimizer/scheduler package is vendored under `LoraEasyCustomOptimizer/` (retains upstream authorship; see that tree's headers).
- **AnimaLoraToolkit** — its web monitor is vendored under `library/monitoring/`. **AnimaLoraToolkit is GPL-3.0** (it derives from ComfyUI's GPL-3.0 model code). The monitor files (`train_monitor.py`, `monitor_smooth.html`) carry no separate license, so they are covered by that GPL-3.0.

⚠️ **Because a GPL-3.0 component is included, this combined work is effectively GPL-3.0.** If you need a more permissive license, replace `library/monitoring/` with an independently-written monitor. See `LICENSE`, `LICENSE-APACHE`, and `NOTICE`. Model weights (Anima / Qwen / VAE) have their own terms — check each model card.
