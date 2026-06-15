# comfyui-anima-trainer

ComfyUI custom node that trains an Anima LoRA from **one image + caption** and
returns a ready-to-use `MODEL`, so a single workflow run covers **curate → train
→ generate**.

Complements `custom_nodes/comfyui-hydralora/` (which only loads already-trained
adapters). Training runs as a **direct subprocess** — preprocess (`tasks.py
preprocess-config`) then `train.py`, spawned from the anima_lora repo root, out
of the ComfyUI process — so a CUDA OOM / segfault kills only the child, not
ComfyUI. The ComfyUI worker blocks until the subprocesses finish (ComfyUI nodes
run synchronously); combined stdout/stderr is teed to the console and a per-run
logfile.

### Install shape

The node must live inside the repo at
`<anima_lora_root>/custom_nodes/comfyui-anima-trainer`: `__init__.py` adds
`<anima_lora_root>` to `sys.path` and locates the workspace root (the dir with
`train.py` + `configs/`) by walking up from this folder. The subprocess is
launched with the same Python interpreter that's running ComfyUI
(`sys.executable`), from the repo root, so the **anima_lora repo and its venv
must be on the same machine** — there's no client/server hop.

## Node

### Anima LoRA Trainer

Inputs:
- `anima_model` — base Anima DiT checkpoint, picked from ComfyUI's
  `diffusion_models` folder (same list as `UNETLoader`). Used as the training
  base **and** reloaded afterwards to build the output MODEL.
- `image` — the IMAGE to train on (single-image mode; each frame in the batch is
  written to a temp dir with the caption).
- `text` — caption for the image.
- `rank` — LoRA rank (`network_dim`); alpha is tied to it.
- `epochs` — number of training epochs.
- `gpu` — hardware tier:
  - `8GB`  → `[low_vram]` preset (gradient checkpointing + unsloth offload)
  - `16GB` → `[default]` preset + `blocks_to_swap=12`
  - `high` → `[32gb]` preset (no swap, no checkpointing)

Output:
- `model` — the chosen base checkpoint loaded and patched with the freshly
  trained LoRA. Use it exactly like the output of the Anima Adapter Loader; no
  separate loader node needed.

The method is locked to `configs/gui-methods/tlora.toml` (T-LoRA + OrthoLoRA).
Saves to
`<ComfyUI>/models/loras/anima_trainer_<timestamp>.safetensors`.

## Notes

- **No MODEL input by design.** Holding the base DiT resident in ComfyUI while
  also spawning a trainer on the same GPU is an easy OOM, so this node does not
  take a MODEL socket. It loads the chosen checkpoint itself *after* training and
  returns it patched.
- **Direct subprocess.** Training runs as two child processes (preprocess then
  `train.py`) spawned from the repo root. No background service is required —
  the node runs everything inline. On a non-zero child exit it raises a `RuntimeError` carrying
  the tail of the captured log.
- **Training is long** (minutes to tens of minutes). The ComfyUI worker blocks
  until the subprocess finishes; hitting Cancel terminates the child and marks
  the node interrupted. Watch the console for per-step logs (also written to
  `<anima_lora_root>/output/tmp_trainer/logs/<output_name>_{preprocess,train}.log`).
- **Memory** — the node calls `comfy.model_management.unload_all_models()` before
  training so there is room for the trainer's own DiT + optimizer state.
- **Single-image mode** writes PNGs + `.txt` under
  `<anima_lora_root>/output/tmp_trainer/<timestamp>/`. Prune periodically.
- **Plain LoRA output** — tlora + ortholora saves as pure LoRA (SVD collapse at
  save time), so the output safetensors is also usable by any ComfyUI LoRA loader.

## Out of scope

- Directory-of-images datasets and method/preset/warm-start overrides (the old
  "Advanced" node) — use the GUI or CLI (`make lora-gui`) for those.
- Baking the trained LoRA into DiT weights as a standalone checkpoint
  (see `scripts/merge_to_dit.py` for the CLI equivalent).
