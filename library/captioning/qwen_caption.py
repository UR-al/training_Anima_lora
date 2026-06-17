# -*- coding: utf-8 -*-
"""Qwen vision-language captioner for the dataset workflow.

Drives the Dataset tab's "Auto-caption (Qwen)" button (via ``tasks.py qwen-caption``,
spawned as a subprocess so torch never touches the GUI). Writes a ``.txt`` caption
beside each image. Two modes (picked in the GUI): ``tags`` (comma-separated booru
tags — feeds the keep-tokens sorter) and ``natural`` (one descriptive sentence); the
prompts live in ``dataset_tags/qwen_caption.toml``.

Layering: the config/prompt helpers are **torch-free** (importable + unit-tested); the
heavy model load + generation is lazy-imported inside :func:`caption_paths`, and the
two model-specific functions — :func:`_load_model` and :func:`_caption_one` — are
isolated so swapping in a different Qwen build is a localized change. The shipped
default targets **Ollama** (``loader = "ollama"`` — Qwen3-VL-8B-NSFW-Caption, served
on the OpenAI-compatible :11434 endpoint, no torch load here); a generic ``openai``
server and the in-process transformers loaders (``qwen2_5_vl`` / ``qwen2_vl``) are
also handled.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

DEFAULT_CONFIG_REL = "dataset_tags/qwen_caption.toml"
VALID_MODES = ("tags", "natural")
# Loaders that speak the OpenAI chat-completions API (no local torch load). Ollama is
# just an OpenAI-compatible server on :11434 — how Qwen3-VL-8B-NSFW-Caption is served.
OPENAI_COMPAT_LOADERS = ("openai", "ollama")


def load_caption_config(path: str | Path) -> dict:
    """Parse the captioner TOML. Raises ``FileNotFoundError`` if absent (the caller
    surfaces a clear "configure the model" message) and ``ValueError`` on bad TOML."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Qwen caption config not found: {p}")
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"bad caption config {p}: {exc}") from exc


def build_prompt(cfg: dict, mode: str) -> str:
    """Instruction for ``mode`` (tags / natural) with the optional {trigger} filled in."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    prompts = cfg.get("prompts") or {}
    prompt = str(prompts.get(mode) or "").strip()
    if not prompt:
        raise ValueError(f"no prompt configured for mode {mode!r} (check [prompts])")
    return prompt.replace("{trigger}", str(cfg.get("trigger") or "").strip())


def _finalize(caption: str, trigger: str) -> str:
    """Tidy raw model text and prepend the trigger word (deduped)."""
    cap = " ".join(caption.split()).strip().strip(",").strip()
    if trigger and not cap.lower().startswith(trigger.lower()):
        cap = f"{trigger}, {cap}" if cap else trigger
    return cap


# --------------------------------------------------------------------------- #
# Model-specific (lazy, heavy) — swap these two for a different Qwen build.
# --------------------------------------------------------------------------- #
def _load_model(cfg: dict):
    """Load (model, processor) for the configured loader. Lazy torch/transformers."""
    import torch
    from transformers import AutoProcessor

    model_path = str(cfg.get("model_path") or "").strip()
    if not model_path:
        raise ValueError(
            "model_path is empty — set it in dataset_tags/qwen_caption.toml"
        )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(
        str(cfg.get("dtype") or "bfloat16"), "auto"
    )
    loader = str(cfg.get("loader") or "qwen2_5_vl").lower()
    if loader == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as Model
    elif loader == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration as Model
    else:
        raise ValueError(f"unsupported loader {loader!r} for local model load")
    model = Model.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.eval()
    return model, AutoProcessor.from_pretrained(model_path)


def _caption_one(model, processor, image, prompt: str, cfg: dict) -> str:
    """Run one image through the VLM → raw caption text (Qwen2-VL/2.5-VL chat API)."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=int(cfg.get("max_new_tokens") or 256)
        )
    trimmed = out[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def ollama_base_url(cfg: dict) -> str:
    """Ollama native-API root. Defaults to http://localhost:11434; tolerates a config
    value carrying an OpenAI-compat ``/v1`` suffix by trimming it. Torch/net-free."""
    base = str(cfg.get("base_url") or "").strip() or "http://localhost:11434"
    base = base.rstrip("/")
    return base[:-3].rstrip("/") if base.endswith("/v1") else base


def ollama_chat_request(
    base_url: str, model_name: str, image_b64: str, prompt: str, cfg: dict
) -> tuple[str, dict]:
    """(url, JSON payload) for Ollama's native /api/chat — vision goes in ``images``
    (raw base64, no data: prefix). Built separately so it's unit-testable without a
    running server. ``num_predict`` is Ollama's max-new-tokens knob."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"num_predict": int(cfg.get("max_new_tokens") or 256)},
    }
    return f"{base_url}/api/chat", payload


def _caption_one_ollama(
    base_url: str, model_name: str, image_b64: str, prompt: str, cfg: dict
) -> str:
    """One image via the local Ollama server (stdlib only — no openai package)."""
    import json
    import urllib.request

    url, payload = ollama_chat_request(base_url, model_name, image_b64, prompt, cfg)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:  # noqa: S310 (local server)
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("message") or {}).get("content", "") or ""


def resolve_openai_endpoint(cfg: dict) -> tuple[str | None, str | None]:
    """(base_url, api_key) for a generic OpenAI-compatible server (loader = 'openai').
    Blank base_url → real api.openai.com; key from cfg else the OPENAI_API_KEY env."""
    base_url = str(cfg.get("base_url") or "").strip()
    return (base_url or None), (str(cfg.get("api_key") or "").strip() or None)


def _build_openai_client(cfg: dict):
    """(client, model_name) for a generic OpenAI-compatible server (loader='openai')."""
    from openai import OpenAI

    base_url, api_key = resolve_openai_endpoint(cfg)
    return OpenAI(base_url=base_url, api_key=api_key), str(
        cfg.get("model_path") or ""
    ).strip()


def _caption_one_openai(
    client, model_name: str, image_b64: str, prompt: str, cfg: dict
):
    """One image via a local OpenAI-compatible server (loader = 'openai')."""
    resp = client.chat.completions.create(
        model=model_name,
        max_tokens=int(cfg.get("max_new_tokens") or 256),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Orchestration (lazy heavy imports inside).
# --------------------------------------------------------------------------- #
def caption_paths(
    paths: list[str | Path],
    mode: str,
    overwrite: bool,
    cfg: dict,
    log=print,
) -> tuple[int, int]:
    """Caption each image path, writing ``<stem>.txt``. Returns (written, skipped).
    Skips images that already have a non-empty caption unless ``overwrite``."""
    from PIL import Image

    prompt = build_prompt(cfg, mode)
    trigger = str(cfg.get("trigger") or "").strip()
    max_side = int(cfg.get("max_image_side") or 0)
    loader = str(cfg.get("loader") or "qwen2_5_vl").lower()

    client = model = processor = base_url = None
    model_name = ""
    if loader == "ollama":
        import base64
        import io

        base_url = ollama_base_url(cfg)
        model_name = str(cfg.get("model_path") or "").strip()
    elif loader == "openai":
        import base64
        import io

        client, model_name = _build_openai_client(cfg)
    else:
        model, processor = _load_model(cfg)

    written = skipped = 0
    for raw in paths:
        img_path = Path(raw)
        txt = img_path.with_suffix(".txt")
        if not overwrite and txt.exists() and txt.read_text(encoding="utf-8").strip():
            skipped += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side))
            if loader in OPENAI_COMPAT_LOADERS:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                if loader == "ollama":
                    raw_cap = _caption_one_ollama(
                        base_url, model_name, b64, prompt, cfg
                    )
                else:
                    raw_cap = _caption_one_openai(client, model_name, b64, prompt, cfg)
            else:
                raw_cap = _caption_one(model, processor, image, prompt, cfg)
            txt.write_text(_finalize(raw_cap, trigger) + "\n", encoding="utf-8")
            written += 1
            log(f"  [{written}] {img_path.name}")
        except Exception as exc:  # noqa: BLE001 — one bad image shouldn't kill the run
            log(f"  ! {img_path.name}: {exc}")
    log(f"done: {written} written, {skipped} skipped")
    return written, skipped
