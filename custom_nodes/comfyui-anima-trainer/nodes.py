"""Anima LoRA Trainer ComfyUI node.

A single, deliberately small node: feed it one IMAGE + a caption, pick the base
Anima checkpoint, a rank, an epoch count and a GPU tier, and on launch it trains
a T-LoRA + OrthoLoRA against that one image/prompt pair.

Design notes:

- **No MODEL input.** Holding the base DiT resident in ComfyUI *and* spawning a
  training subprocess on the same GPU is an easy OOM. So this node does not take
  a MODEL socket; instead it loads the chosen Anima checkpoint itself **after**
  training finishes and returns a patched MODEL — a drop-in for
  ``UNETLoader → Anima Adapter Loader``.
- **Direct subprocess.** Training runs as a plain ``subprocess`` (preprocess
  then ``train.py``) spawned from the anima_lora repo root, out of the ComfyUI
  process, so a CUDA OOM / segfault kills only the child — not ComfyUI. The
  ComfyUI worker blocks until it finishes (ComfyUI nodes run synchronously).
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys

import folder_paths  # ComfyUI builtin

# Training deps are imported lazily inside `train()` so that merely loading this
# module in ComfyUI doesn't force `library.*` imports (which pull torch
# extensions and slow startup).


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _anima_lora_root() -> str:
    from .training import find_anima_lora_root

    return find_anima_lora_root(os.path.dirname(__file__))


def _load_node_defaults() -> dict:
    """Read the sibling ``node_defaults.toml`` of trainer-node-only overrides.

    These tune the few-image training regime this node runs in (DataLoader
    worker policy, log cadence) without editing the shared ``configs/base.toml``.
    They're layered *below* the UI fields in ``train()`` so user choices win.
    Re-read on every run; a missing or unparseable file is treated as empty so
    the node still works if it's deleted.
    """
    import tomllib

    path = os.path.join(os.path.dirname(__file__), "node_defaults.toml")
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(
            f"[Anima Trainer] could not read node_defaults.toml ({e}); "
            f"using base/preset defaults.",
            flush=True,
        )
        return {}


def _input_subdirs() -> list[str]:
    """List immediate subdirectories of ComfyUI's input dir, as relative names.

    Populates the folder-mode trainer's dataset dropdown. Re-evaluated whenever
    ComfyUI rebuilds INPUT_TYPES (graph load / refresh), so newly-added dataset
    folders show up after a node refresh. Returns ``[""]`` (a single blank entry)
    when the input dir has no subfolders, so the node still loads.
    """
    try:
        root = folder_paths.get_input_directory()
        subdirs = sorted(
            entry
            for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        )
    except OSError:
        subdirs = []
    return subdirs or [""]


_MASK_NONE = "(none)"


def _input_subdirs_optional() -> list[str]:
    """Input-dir subfolders prefixed with a ``(none)`` sentinel for the mask picker."""
    subdirs = [d for d in _input_subdirs() if d]
    return [_MASK_NONE, *subdirs]


def _comfy_loras_dir() -> str:
    """Return the directory to save trained LoRAs into, under ComfyUI's loras.

    Prefers ComfyUI's *native* ``models/loras`` over any path an
    ``extra_model_paths.yaml`` entry registers (which, with ``is_default: true``,
    would otherwise sort first — e.g. anima_lora's ``output/``). Falls back to
    the first registered loras path when the native dir isn't registered.
    """
    paths = folder_paths.get_folder_paths("loras") or []
    if not paths:
        raise RuntimeError("ComfyUI has no 'loras' folder registered.")
    models_dir = getattr(folder_paths, "models_dir", None)
    if models_dir:
        native = os.path.abspath(os.path.join(models_dir, "loras"))
        for p in paths:
            if os.path.abspath(p) == native:
                return p
    return paths[0]


def _comfy_model_path(folder: str, preferred: str) -> str | None:
    """Resolve a model file through ComfyUI's ``folder_paths``, or ``None``.

    Lets the trainer source the VAE / text-encoder used for caching from
    whatever ComfyUI registers (its ``models/`` dirs, or any ``base_path`` from
    ``extra_model_paths.yaml``) — the same way the base DiT is already resolved —
    instead of assuming a copy under ``anima_lora/models/``. Returns ``None``
    when nothing usable is found, so the caller falls back to the
    preprocess-config defaults rather than passing a bad path.

    Tries the canonical Anima filename first; if absent, accepts a sole file in
    the folder (the common case — one VAE, one TE); otherwise gives up rather
    than guess among several.
    """
    path = folder_paths.get_full_path(folder, preferred)
    if path:
        return path
    files = folder_paths.get_filename_list(folder) or []
    if len(files) == 1:
        return folder_paths.get_full_path(folder, files[0])
    return None


def _overrides_to_argv(overrides: dict) -> list[str]:
    """Flatten an ``overrides`` dict into ``--key value`` ``train.py`` argv.

    Mirrors ``scripts/tasks/training.py::_toml_table_to_argv``: bools become a
    bare ``--flag`` when true (omitted when false), lists/tuples spread into
    ``--key v1 v2``, scalars become ``--key str(value)``. These are appended to
    the ``train.py`` command after ``--method``/``--preset``/``--methods_subdir``
    so the CLI-overrides-win merge chain applies them on top of the gui-method
    config — exactly the precedence chained CLI overrides have.
    """
    argv: list[str] = []
    for key, val in overrides.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.append(flag)
            argv.append(str(val))
    return argv


def _run_subprocess(argv: list[str], *, root: str, log_path: str, label: str) -> None:
    """Run ``argv`` from ``root`` to completion, teeing output to ``log_path``.

    Streams the child's combined stdout/stderr both to the ComfyUI console (so
    per-step logs show live) and to ``log_path``. Polls ComfyUI's interrupt flag
    each line and, on Cancel, terminates the child and raises
    ``InterruptProcessingException`` so the node is marked cancelled. On a
    non-zero exit, raises ``RuntimeError`` carrying the tail of the captured log
    (so a failed run surfaces its stdout tail directly in the ComfyUI error).
    """
    import comfy.model_management

    # Inherit the parent env; ensure unbuffered child stdio so the live console
    # tail isn't chunked, and that the venv's bin dir is on PATH for any
    # console-script grandchildren the trainer spawns.
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    venv_bin = os.path.dirname(sys.executable)
    if venv_bin and venv_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    print(f"[Anima Trainer] {label}: {' '.join(argv)}", flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    tail: list[str] = []
    with open(log_path, "w", encoding="utf-8", errors="replace") as logfh:
        proc = subprocess.Popen(
            argv,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                logfh.write(line)
                logfh.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
                tail.append(line)
                if len(tail) > 40:
                    del tail[0]
                if comfy.model_management.processing_interrupted():
                    print(
                        f"[Anima Trainer] interrupted — terminating {label} "
                        "subprocess",
                        flush=True,
                    )
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise comfy.model_management.InterruptProcessingException()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(
            f"{label} subprocess failed (exit {rc}). See the log: {log_path}\n"
            f"--- last lines ---\n{''.join(tail).rstrip()}"
        )


def _trainer_tmp_root() -> str:
    """Where single-image-mode datasets are staged before training.

    Prefers the repo's ``output/tmp_trainer`` (so it's covered by the repo
    gitignore and easy to prune), but falls back to ComfyUI's temp dir when the
    node is installed outside the anima_lora tree — the training subprocess
    reads it over a plain filesystem path either way (same machine).
    """
    try:
        return os.path.join(_anima_lora_root(), "output", "tmp_trainer")
    except Exception:
        return os.path.join(folder_paths.get_temp_directory(), "anima_trainer")


# ---------------------------------------------------------------------------
# Train via direct subprocess → block until done → return saved safetensors path
# ---------------------------------------------------------------------------


def _train_and_save(
    *,
    method: str,
    preset: str,
    overrides: dict,
    image=None,
    prompt: str = "",
    dataset_dir: str = "",
    mask=None,
    mask_dir: str = "",
) -> str:
    """Train in a direct subprocess and block until done.

    Either an ``image`` + ``prompt`` (single-image mode) or a ``dataset_dir``
    (directory mode) supplies the data; ``prepare_dataset_dir`` picks the mode.
    An optional ``mask`` (MASK tensor, single-image) or ``mask_dir`` (directory)
    turns on masked loss.

    Spawns two subprocesses from the anima_lora repo root, out of the ComfyUI
    process (so a CUDA OOM / segfault kills only the child): first
    ``tasks.py preprocess-config …`` (bucket-resize + VAE/TE cache the dataset),
    then ``train.py --method tlora --preset … --methods_subdir gui-methods …``
    with the UI fields folded in as CLI overrides. Blocks until each finishes,
    raising ``RuntimeError`` with the log tail on a non-zero exit, then returns
    the absolute path of the saved safetensors.
    """
    import comfy.model_management

    from .dataset_prep import prepare_dataset_dir

    root = _anima_lora_root()

    # image set → single-image mode (writes the IMAGE batch + caption sidecars);
    # dataset_dir set → directory mode (user's dir of images + .txt sidecars).
    # src_dir = originals (read-only input to preprocess); image_dir/cache_dir =
    # where resized images + caches land; dataset_cfg names image_dir + cache_dir.
    (
        src_dir,
        _image_dir,
        _cache_dir,
        dataset_cfg,
        n_images,
        resolved_mask_dir,
    ) = prepare_dataset_dir(
        image,
        prompt,
        dataset_dir,
        tmp_root=_trainer_tmp_root(),
        mask=mask,
        mask_dir=mask_dir,
    )

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"anima_trainer_{ts}"
    output_dir = _comfy_loras_dir()

    overrides = dict(overrides)
    overrides.setdefault("dataset_config", dataset_cfg)
    overrides.setdefault("output_dir", output_dir)
    overrides.setdefault("output_name", output_name)
    # Masks present → force masked loss on (the gui-method TOML leaves it off).
    # The dataset config already carries the resolved mask_dir.
    if resolved_mask_dir:
        overrides["masked_loss"] = True

    print(
        f"[Anima Trainer] training method={method} preset={preset} "
        f"images={n_images}{' +masks' if resolved_mask_dir else ''} "
        f"→ {output_name}.safetensors",
        flush=True,
    )

    # Free ComfyUI-held VRAM so the (separate) training process has room for its
    # own DiT + optimizer state. The child subprocesses are the only GPU users.
    comfy.model_management.unload_all_models()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    log_dir = os.path.join(_trainer_tmp_root(), "logs")

    # train.py refuses to run on an incomplete latent/TE cache, and the temp
    # dir starts empty. So phase 1 runs `tasks.py preprocess-config` as a direct
    # subprocess that bucket-resizes + caches the dataset; phase 2 then runs
    # `train.py` once the caches exist. Both block inline and run serially in
    # their own process, so a CUDA OOM kills only the child — and they never
    # fight over VRAM (the trainer subprocess starts after preprocess exits).
    # Cache against the models ComfyUI registers (the DiT the user selected, plus
    # the VAE + text-encoder resolved through folder_paths) so preprocess never
    # assumes a copy under anima_lora/models/. Unresolved ones are simply omitted
    # → preprocess-config falls back to its config-default models/ paths.
    pp_argv = [
        sys.executable,
        "tasks.py",
        "preprocess-config",
        "--dataset_config",
        dataset_cfg,
        "--src",
        src_dir,
    ]
    dit_path = overrides.get("pretrained_model_name_or_path")
    if dit_path:
        pp_argv += ["--dit", dit_path]
    vae_path = _comfy_model_path("vae", "qwen_image_vae.safetensors")
    if vae_path:
        pp_argv += ["--vae", vae_path]
    qwen3_path = _comfy_model_path("text_encoders", "qwen_3_06b_base.safetensors")
    if qwen3_path:
        pp_argv += ["--qwen3", qwen3_path]

    # Phase 1: preprocess (bucket-resize + VAE/TE cache the dataset).
    _run_subprocess(
        pp_argv,
        root=root,
        log_path=os.path.join(log_dir, f"{output_name}_preprocess.log"),
        label="preprocess",
    )

    # Phase 2: training. Same CLI surface the `make lora-gui` path builds —
    # `train.py --method <m> --preset <p> --methods_subdir gui-methods` with the
    # UI fields folded in as `--key value` CLI overrides (CLI wins the merge
    # chain, exactly the precedence the gui-method config expects).
    train_argv = [
        sys.executable,
        "train.py",
        "--method",
        method,
        "--preset",
        preset,
        "--methods_subdir",
        "gui-methods",
    ]
    train_argv += _overrides_to_argv(overrides)

    _run_subprocess(
        train_argv,
        root=root,
        log_path=os.path.join(log_dir, f"{output_name}_train.log"),
        label="train",
    )

    # train.py writes `<output_dir>/<output_name>.safetensors` at the end; we set
    # both above, so the path is deterministic.
    expected = os.path.join(output_dir, f"{output_name}.safetensors")
    if not os.path.exists(expected):
        raise RuntimeError(
            f"Training finished but no checkpoint found (expected {expected}). "
            f"See the train log: {os.path.join(log_dir, output_name + '_train.log')}"
        )
    print(f"[Anima Trainer] saved {expected}", flush=True)
    return expected


def _load_anima_model(anima_model: str):
    """Load a base Anima DiT from ComfyUI's diffusion_models folder as a MODEL.

    Replicates ComfyUI's ``UNETLoader.load_unet`` (default weight dtype) so the
    returned object is a normal ``ModelPatcher`` — identical to what the user
    would get by chaining a UNETLoader.
    """
    import comfy.sd

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", anima_model)
    return comfy.sd.load_diffusion_model(unet_path, model_options={})


def _apply_lora_to_model(model, file_path: str, strength: float) -> None:
    """Patch the trained LoRA onto ``model`` in place via ComfyUI's native path.

    The trainer only ever emits an ortho-T-LoRA, which serialises as plain LoRA
    keys: OrthoLoRA folds down to ``lora_down``/``lora_up`` at save time, and the
    T-LoRA rank mask is training-only (inference is full-rank). So ComfyUI's
    stock machinery — ``model_lora_keys_unet`` + ``convert_lora`` + ``load_lora``
    — maps and applies it directly, exactly as the built-in LoraLoader node would
    on a natively-loaded Anima DiT. No Anima adapter loader (HydraLoRA /
    Chimera live routing) is involved, so we don't pull in the comfyui-hydralora
    node here (which imports its chimera module at load time).
    """
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils

    lora_sd = comfy.utils.load_torch_file(file_path, safe_load=True)
    lora_sd = _fold_inv_scale(lora_sd)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    lora_sd = comfy.lora_convert.convert_lora(lora_sd)
    loaded = comfy.lora.load_lora(lora_sd, key_map)
    model.add_patches(loaded, strength)


def _fold_inv_scale(lora_sd: dict) -> dict:
    """Fold ``per_channel_scaling`` ``inv_scale`` into ``lora_down`` and drop it.

    Inert for the trainer's default ortho-T-LoRA (no ``per_channel_scaling`` →
    no ``.inv_scale`` keys). Kept as a guard so the native patcher never silently
    drops an ``.inv_scale`` suffix it doesn't recognise and applies a delta
    that's off by ``s_norm`` per input column. Mirrors ``LoRAModule.merge_to``:
    ``down *= inv_scale`` then strip the key. Returns a new dict.
    """
    inv_keys = [k for k in lora_sd if k.endswith(".inv_scale")]
    if not inv_keys:
        return lora_sd
    import torch

    out = dict(lora_sd)
    for inv_key in inv_keys:
        down_key = f"{inv_key[: -len('.inv_scale')]}.lora_down.weight"
        inv_scale = out.pop(inv_key)
        down = out.get(down_key)
        if down is None or down.dim() != 2:
            continue
        out[down_key] = down.to(torch.float) * inv_scale.to(torch.float).unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class AnimaLoRATrainer:
    """Train an Anima LoRA (T-LoRA + OrthoLoRA) from one image + caption.

    Loads the chosen base checkpoint itself after training and returns a patched
    MODEL — use it exactly like the output of an Anima Adapter Loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models") or [""]
        return {
            "required": {
                "anima_model": (
                    unets,
                    {
                        "tooltip": (
                            "Base Anima DiT checkpoint (ComfyUI diffusion_models "
                            "folder). Used for training and reloaded afterwards to "
                            "produce the output MODEL."
                        )
                    },
                ),
                "image": ("IMAGE", {"tooltip": "The image(s) to train on."}),
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Caption for the training image (STRING input).",
                    },
                ),
                "save_as": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Output LoRA filename (without extension), saved into "
                            "ComfyUI's loras folder. Leave blank for an "
                            "auto-timestamped name."
                        ),
                    },
                ),
                "rank": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "tooltip": "LoRA rank (network_dim); alpha is tied to it.",
                    },
                ),
                "epochs": (
                    "INT",
                    {"default": 25, "min": 1, "max": 10000},
                ),
                "lr": (
                    "FLOAT",
                    {
                        "default": 5e-5,
                        "min": 1e-7,
                        "max": 1e-2,
                        "step": 1e-6,
                        "round": False,
                        "tooltip": "Learning rate (network learning_rate).",
                    },
                ),
                "gpu": (["8GB", "16GB", "high"], {"default": "16GB"}),
            },
            "optional": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Optional loss mask(s). White = train on this region, "
                            "black = ignore. One mask per image, or a single mask "
                            "broadcast to all. Connecting it turns on masked loss."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "train"
    CATEGORY = "anima/training"
    DESCRIPTION = (
        "Train an Anima T-LoRA + OrthoLoRA from a single image + caption, then "
        "load the chosen base checkpoint and return it with the trained LoRA "
        "applied — a drop-in for the Anima Adapter Loader's MODEL output. "
        "Optionally accepts a MASK to train with masked loss. Training runs as a "
        "direct subprocess (preprocess then train.py) spawned from the anima_lora "
        "repo root; the ComfyUI worker blocks until it finishes. Watch the console "
        "for per-step logs."
    )

    def train(
        self,
        anima_model: str,
        image,
        text: str,
        rank: int,
        epochs: int,
        lr: float,
        save_as: str,
        gpu: str,
        mask=None,
    ):
        from .training import GPU_TIER_PRESET

        anima_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", anima_model
        )

        overrides: dict = {
            "network_dim": int(rank),
            "network_alpha": float(rank),
            "max_train_epochs": int(epochs),
            "learning_rate": float(lr),
            # Train against the user-selected base checkpoint.
            "pretrained_model_name_or_path": anima_path,
        }
        # Layer the node-only training defaults *below* the explicit UI fields
        # already in `overrides` (rank/epochs/lr/model) — setdefault means
        # anything the user typed in the node still wins. The gpu dropdown only
        # selects the hardware preset (see GPU_TIER_PRESET); blocks_to_swap and
        # the checkpointing flags come from that preset + node_defaults.toml.
        for key, value in _load_node_defaults().items():
            overrides.setdefault(key, value)

        # Optional user-supplied output name; blank → auto-timestamped default in
        # `_train_and_save`. Strip any path parts / extension so it can't escape
        # the loras folder or end up double-suffixed (`x.safetensors.safetensors`).
        name = os.path.splitext(os.path.basename(save_as.strip()))[0]
        if name:
            overrides["output_name"] = name

        saved_path = _train_and_save(
            method="tlora",
            preset=GPU_TIER_PRESET[gpu],
            overrides=overrides,
            image=image,
            prompt=text,
            mask=mask,
        )

        # Load the base DiT ourselves (avoids holding it resident during the run)
        # and return it patched — a drop-in for the Anima Adapter Loader.
        model = _load_anima_model(anima_model)
        _apply_lora_to_model(model, saved_path, 1.0)
        return (model,)


class AnimaLoRATrainerFolder:
    """Train an Anima LoRA (T-LoRA + OrthoLoRA) from a folder of images + captions.

    Like ``AnimaLoRATrainer`` but reads its dataset from a directory of images
    each paired with a same-stem ``.txt`` caption sidecar, instead of a single
    connected IMAGE. Loads the chosen base checkpoint after training and returns
    a patched MODEL — a drop-in for the Anima Adapter Loader's MODEL output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        unets = folder_paths.get_filename_list("diffusion_models") or [""]
        return {
            "required": {
                "anima_model": (
                    unets,
                    {
                        "tooltip": (
                            "Base Anima DiT checkpoint (ComfyUI diffusion_models "
                            "folder). Used for training and reloaded afterwards to "
                            "produce the output MODEL."
                        )
                    },
                ),
                "dataset_dir": (
                    _input_subdirs(),
                    {
                        "tooltip": (
                            "Subfolder of ComfyUI's input/ directory holding the "
                            "training images, each with a same-stem .txt caption "
                            "sidecar next to it. Read-only (images are "
                            "bucket-resized into a temp dir). Refresh the node to "
                            "pick up newly-added folders."
                        ),
                    },
                ),
                "save_as": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Output LoRA filename (without extension), saved into "
                            "ComfyUI's loras folder. Leave blank for an "
                            "auto-timestamped name."
                        ),
                    },
                ),
                "rank": (
                    "INT",
                    {
                        "default": 16,
                        "min": 1,
                        "max": 256,
                        "tooltip": "LoRA rank (network_dim); alpha is tied to it.",
                    },
                ),
                "epochs": (
                    "INT",
                    {"default": 25, "min": 1, "max": 10000},
                ),
                "lr": (
                    "FLOAT",
                    {
                        "default": 5e-5,
                        "min": 1e-7,
                        "max": 1e-2,
                        "step": 1e-6,
                        "round": False,
                        "tooltip": "Learning rate (network learning_rate).",
                    },
                ),
                "gpu": (["8GB", "16GB", "high"], {"default": "16GB"}),
            },
            "optional": {
                "mask_dir": (
                    _input_subdirs_optional(),
                    {
                        "tooltip": (
                            "Optional subfolder of ComfyUI's input/ directory "
                            "holding `{stem}_mask.png` loss masks (white = keep), "
                            "one per training image by matching stem. Pick "
                            "'(none)' to train without masked loss."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "train"
    CATEGORY = "anima/training"
    DESCRIPTION = (
        "Train an Anima T-LoRA + OrthoLoRA from a folder of images + caption "
        "sidecars, then load the chosen base checkpoint and return it with the "
        "trained LoRA applied — a drop-in for the Anima Adapter Loader's MODEL "
        "output. Optionally point it at a folder of `{stem}_mask.png` masks to "
        "train with masked loss. Training runs as a direct subprocess (preprocess "
        "then train.py) spawned from the anima_lora repo root; the ComfyUI worker "
        "blocks until it finishes. Watch the console for per-step logs."
    )

    def train(
        self,
        anima_model: str,
        dataset_dir: str,
        rank: int,
        epochs: int,
        lr: float,
        save_as: str,
        gpu: str,
        mask_dir: str = _MASK_NONE,
    ):
        from .training import GPU_TIER_PRESET

        anima_path = folder_paths.get_full_path_or_raise(
            "diffusion_models", anima_model
        )

        # Resolve the chosen subfolder name to an absolute path under ComfyUI's
        # input dir. `get_input_directory` honours the same root the dropdown was
        # built from in `_input_subdirs`.
        if not dataset_dir:
            raise ValueError(
                "No dataset folder selected. Drop a folder of images + .txt "
                "captions into ComfyUI's input/ directory and refresh the node."
            )
        dataset_path = os.path.join(folder_paths.get_input_directory(), dataset_dir)

        # Optional mask folder, resolved the same way; "(none)" → no masked loss.
        mask_path = ""
        if mask_dir and mask_dir != _MASK_NONE:
            mask_path = os.path.join(folder_paths.get_input_directory(), mask_dir)

        overrides: dict = {
            "network_dim": int(rank),
            "network_alpha": float(rank),
            "max_train_epochs": int(epochs),
            "learning_rate": float(lr),
            # Train against the user-selected base checkpoint.
            "pretrained_model_name_or_path": anima_path,
        }
        # Layer the node-only training defaults *below* the explicit UI fields
        # already in `overrides` (rank/epochs/lr/model) — setdefault means
        # anything the user typed in the node still wins. The gpu dropdown only
        # selects the hardware preset (see GPU_TIER_PRESET); blocks_to_swap and
        # the checkpointing flags come from that preset + node_defaults.toml.
        for key, value in _load_node_defaults().items():
            overrides.setdefault(key, value)

        # Optional user-supplied output name; blank → auto-timestamped default in
        # `_train_and_save`. Strip any path parts / extension so it can't escape
        # the loras folder or end up double-suffixed (`x.safetensors.safetensors`).
        name = os.path.splitext(os.path.basename(save_as.strip()))[0]
        if name:
            overrides["output_name"] = name

        saved_path = _train_and_save(
            method="tlora",
            preset=GPU_TIER_PRESET[gpu],
            overrides=overrides,
            dataset_dir=dataset_path,
            mask_dir=mask_path,
        )

        # Load the base DiT ourselves (avoids holding it resident during the run)
        # and return it patched — a drop-in for the Anima Adapter Loader.
        model = _load_anima_model(anima_model)
        _apply_lora_to_model(model, saved_path, 1.0)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "AnimaLoRATrainer": AnimaLoRATrainer,
    "AnimaLoRATrainerFolder": AnimaLoRATrainerFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaLoRATrainer": "Anima LoRA Trainer",
    "AnimaLoRATrainerFolder": "Anima LoRA Trainer (Folder)",
}
