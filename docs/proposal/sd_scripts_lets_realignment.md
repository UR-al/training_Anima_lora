# sd-scripts / LETS realignment — target structure proposal (v3, CONFIRMED)

Status: **CONFIRMED** (structure agreed; execution proceeds in phases).
Date: 2026-06-15. Supersedes v1/v2.

## The blueprint: a 4-donor layered fusion (with pinned references)

| Layer | Donor reference | Style we adopt |
|---|---|---|
| **GUI** | `Source2Spy/kohya_ss_anima` (kohya_ss, anima-aware) | kohya `kohya_gui/` LoRA + Utilities tab frame |
| **Folder structure** | `kohya-ss/sd-scripts` + `67372a/LoRA_Easy_Training_Scripts` | mirror their layout — **do not invent names** |
| **Training config** | sd-scripts + LETS | one plain TOML (`network_module` + `network_args` carry routing), `save_toml`/`load_toml`, runnable as `train.py --config_file …` |
| **Training engine** | `sorryhyun/anima_lora` | **this repo** — `train.py` + `library/` + `networks/` |
| **Monitoring** | `Moeblack/AnimaLoraToolkit` | already merged → `library/monitoring/` (`--monitor`, separate port) |

## Reference layouts (cloned + inspected 2026-06-15)

**kohya/sd-scripts** (engine): root train scripts + `library/`, `networks/`, `tools/`, `finetune/`, `configs/`, `docs/`, `tests/`.

**LETS frontend** (`67372a/LoRA_Easy_Training_Scripts`): `main.py`, `ui_files/` (incl. `AnimaUI.py` — already anima-aware!), `modules/` (`TomlFunctions.py` = config save/load, `NetworkManager.py`, `OptimizerItem.py`, `QueueItem.py`), `css/`, `icons/`.

**LETS backend** (`…_Backend`): `main.py` (FastAPI), `sd_scripts/` (submodule), **`custom_scheduler/LoraEasyCustomOptimizer/`** (the optimizer zoo lives here), `utils/`.

**Key takeaway:** mirroring these means we do **not** invent names — `library/`,
`networks/`, `tools/`, `finetune/` are the sd-scripts names; the optimizer zoo's
"proper" home is `custom_scheduler/LoraEasyCustomOptimizer/` (LETS). Keeping
`library/`+`networks/` importable **at root** is also what preserves
`network_module = "networks.lora_anima"` and LETS-config compatibility — moving
them under a `backend/` package would break every config's dotted path.

## Target structure for this repo

```
training_Anima_lora/
│  train.py, inference.py          # engine entry (sd-scripts root scripts)        [keep]
│  tasks.py, Makefile, pyproject.toml
├── library/                       # sd-scripts core (anima_lora engine)           [keep name]
│   ├── api/                       # ← anima_lora/ façade folded here              [MOVE]
│   └── monitoring/                # AnimaLoraToolkit monitor                       [keep]
├── networks/                      # sd-scripts adapters; importlib create_network [keep name]
├── tools/                         # ← sd-scripts-style utils                       [NEW]
│                                  #   from scripts/preprocess (cache_latents / _te),
│                                  #   scripts/merge_to_dit.py, show_metadata, resize
├── finetune/                      # ← sd-scripts-style captioning/tagging          [NEW]
│                                  #   from scripts/anima_tagger/, captioning
├── custom_scheduler/
│   └── LoraEasyCustomOptimizer/   # ← optimizer zoo, LETS placement                [MOVE]
├── gui/                           # ← LETS/kohya frontend                          [NEW]
│   ├── kohya/                     #   vendored kohya LoRA + Utilities tabs
│   ├── webgui/                    #   (moved from scripts/webgui)
│   └── modules/                   #   config save/load (TomlFunctions-equiv) + launcher
├── configs/                       # sd-scripts configs + LETS example TOMLs        [keep]
│   └── examples/                  #   ← pure --config_file runnable samples         [NEW]
├── bench/, docs/, custom_nodes/   # [keep]
└── scripts/                       # orchestration left over (tasks.py bodies)      [shrinks]
```

### Migration mapping (current → target)

| Current | Target | Note |
|---|---|---|
| `anima_lora/` | `library/api/` | re-export shim at old path during transition |
| `LoraEasyCustomOptimizer/` | `custom_scheduler/LoraEasyCustomOptimizer/` | resolver in `library/training/optimizers.py` + `pyproject` updated; shim keeps friendly-name registry |
| `scripts/webgui/`, `scripts/gradio_gui/` | `gui/webgui/`, `gui/kohya/` | GUI consolidates |
| `scripts/preprocess/` (cache utils) | `tools/` | sd-scripts `tools/` |
| `scripts/anima_tagger/`, captioning | `finetune/` | sd-scripts `finetune/` |
| `scripts/daemon/` | **DELETED** (done 2026-06-15) | training is inline-only; GUI + ComfyUI node `Popen` `train.py` directly |
| `library/`, `networks/`, `train.py` | **unchanged at root** | = sd-scripts layout; preserves dotted `network_module` + LETS config compat |

> **On "전면 rename":** the reference repos you chose (sd-scripts, LETS) *use*
> `library/`/`networks/`. Mirroring them = keep those names. The "clean
> separation" you want is achieved by **relocation** (optimizer → custom_scheduler/,
> façade → library/api/, GUI → gui/, utils → tools/+finetune/), not by renaming the
> two standard packages. If you still want different top-level names for
> `library`/`networks`, say so — but it diverges from the references and breaks
> config dotted paths. **Need your call (§decision).**

## Config model (sd-scripts/LETS)

Primary CLI: `python train.py --config_file run.toml` (already works — lenient
loader, routing via `network_args`). The GUI's `modules/` writes this TOML
(`save_toml`) exactly like LETS `TomlFunctions`. `--method`/`--preset` stay as an
optional convenience layer. Ship `configs/examples/*.toml` (pure-config runnable).

## Daemon — FULL removal (confirmed)

The daemon is **removed outright**, not just from the GUI flow:
- delete `scripts/daemon/`;
- drop `--queue` from `scripts/tasks/_common.py` (inline `build_launch_cmd` path
  becomes the only launch);
- switch `scripts/webgui/server.py` `launch()` to the direct `Popen` path (it
  currently defaults to daemon-submit);
- the ComfyUI trainer node (`custom_nodes/comfyui-anima-trainer/`) loses its
  backend — rewire to a direct subprocess or drop the node;
- remove the `server.log_tail` daemon tail + the gradio terminal-mirror panel
  added earlier.

GUI runs `python train.py --config_file …` directly; logs go to the terminal;
monitoring is the separate `--monitor` port.

## Confirmed decisions

1. **Names = mirror sd-scripts/LETS** — keep `library/`/`networks/` at root
   (they are the reference names; preserves dotted `network_module` + LETS config
   compat). No custom renames.
2. **`scripts/` reorg**: move only the sd-scripts-equivalent utilities to
   `tools/`+`finetune/`; orchestration (`scripts/tasks/`) stays (it backs `make`).
3. **Daemon**: full removal (above).

## Phasing (after approval)

- **P1 — config + launch (no moves):** bless `--config_file` LETS path; add
  `configs/examples/*.toml`; switch GUI/launch to direct CLI (drop daemon); docs.
- **P2 — relocation:** introduce `gui/`, `tools/`, `finetune/`, `custom_scheduler/`,
  `library/api/`; move with **re-export shims** at old paths; update `pyproject`
  `packages.find`; `make test-unit` green each step.
- **P3 — GUI:** vendor kohya LoRA + Utilities into `gui/kohya/`; rewire Train to
  emit a pure config + `python train.py --config_file …`; add Anima toggles +
  `--monitor`.
```
