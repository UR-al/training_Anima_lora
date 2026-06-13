# Anima LoRA — merged trainer

A fast **Anima** LoRA trainer that merges three projects into one:

- **[anima_lora](https://github.com/sorryhyun/anima_lora)** — the base: `torch.compile`-accelerated LoRA training of the Anima DiT (constant-token bucketing + native-flatten compile, flash-attn, block-swap, fully-cached dataloader).
- **LoRA_Easy_Training_Scripts** — a broad optimizer/scheduler suite (~89 optimizers + custom LR schedulers), vendored as `LoraEasyCustomOptimizer/`.
- **AnimaLoraToolkit** — a live, dependency-free web monitor (loss/LR/sample dashboard), vendored under `library/monitoring/`.

## Highlights

- **Fast training** — native `torch.compile` path (~1.3 s/it at rank 32, 1 MP on a 16 GB card).
- **~89 optimizers by name** — `--optimizer_type CAME` (or `ADOPT`, `Prodigy`, `ProdigyPlusScheduleFree`, `FMARSCrop`, `OCGOpt`, …, or any `pkg.module.Class`). kohya built-ins (`AdamW` fused, `AdamW8bit`, `DAdapt*`, `Adafactor`, `*ScheduleFree`) still work. Meta-optimizers wrap a base via `--optimizer_args base_optimizer_type='CAME'`. Missing optional deps are skipped, never fatal.
- **Custom LR schedulers** — `--lr_scheduler_type <dotted path>` (e.g. CAWR / RAWR) with `--lr_scheduler_args`.
- **Live web monitor** — `--monitor` serves a Chart.js loss/LR/sample dashboard; resumes the curve on `--resume`.
- **Web control panel** — `run_gui.bat` (or `python tasks.py webgui`): configure → launch → monitor in the browser, no Qt.

## Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 (8 GB) | 16 GB+ |
| System RAM | 16 GB | 32 GB+ |
| Disk | 60 GB | 200 GB+ (caches + outputs) |
| Stack | Windows 11 / Ubuntu 22.04+, NVIDIA driver ≥595 | CUDA 13.2 toolkit (for `torch.compile`/Triton) |

Python 3.13 + PyTorch 2.12 (cu132) are installed for you by the installer.

## Install (Windows)

Clone, then double-click one of:

- **`install.bat`** — via **uv** (recommended; exact locked deps).
- **`install_pip.bat`** — via **pip** (needs Python 3.13 on PATH).

```powershell
git clone https://github.com/UR-al/training_Anima_lora
cd training_Anima_lora
install.bat
```

Then fetch the model weights (Anima DiT + Qwen3 text encoder + VAE) into `models/` (gitignored — not shipped in this repo).

## Use

```powershell
run_gui.bat                                            REM web control panel in the browser
python tasks.py lora --optimizer_type CAME --monitor   REM CLI: train + live dashboard
update.bat                                             REM git pull + re-sync deps
```

`python tasks.py lora --method <m> --preset <p> [--dataset_config x.toml] [overrides]` is the core entry; `make help` / the `COMMANDS` table in `tasks.py` lists every target. See `CLAUDE.md` for the architecture and the full merged-capabilities reference.

## License & attribution

This is a derivative of the three projects above; see `LICENSE`, `LICENSE-APACHE`, and `NOTICE`. The vendored `LoraEasyCustomOptimizer/` optimizers and `library/monitoring/` dashboard retain their upstream authorship.
