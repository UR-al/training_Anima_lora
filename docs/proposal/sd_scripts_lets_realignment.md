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
│   ├── api/                       # ← anima_lora/ façade folded here              [DONE]
│   └── monitoring/                # AnimaLoraToolkit monitor                       [keep]
├── networks/                      # sd-scripts adapters; importlib create_network [keep name]
├── tools/                         # ← sd-scripts-style utils                       [NEW]
│                                  #   from scripts/preprocess (cache_latents / _te),
│                                  #   scripts/merge_to_dit.py, show_metadata, resize
├── finetune/                      # ← sd-scripts-style captioning/tagging          [NEW]
│                                  #   from scripts/anima_tagger/, captioning
├── custom_scheduler/
│   └── LoraEasyCustomOptimizer/   # ← optimizer zoo, LETS placement                [DONE]
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

| Current | Target | Status / Note |
|---|---|---|
| `scripts/webgui/`, `scripts/gradio_gui/` | `gui/backend.py`, `gui/kohya/` | **DONE 2026-06-15** — GUI consolidated under `gui/`; the stdlib HTTP web GUI was then **removed** (kohya Gradio is the only GUI), its shared command-builder/launcher backend kept as `gui/backend.py` |
| `scripts/anima_tagger/` | `finetune/anima_tagger/` | **DONE 2026-06-15** — captioning *logic* stays in `library.captioning` |
| `scripts/preprocess/`, `scripts/merge_to_dit.py` | `tools/` | **DONE 2026-06-15** — flat sd-scripts `tools/`, run by path |
| `scripts/daemon/` | **DELETED** | done 2026-06-15; training inline-only |
| `anima_lora/` | `library/api/` | **DONE 2026-06-15** — façade moved to `library/api/` (`ROOT` = parents[2]); `anima_lora/` kept as a lazy PEP-562 shim delegating to `library.api` so `import anima_lora` still works. Verified torch-free (import + ROOT + façade `dir()` + lazy dispatch). `library*` find-include auto-discovers `library.api`; **local `uv sync` re-sync still recommended** before an embedder run. |
| `LoraEasyCustomOptimizer/` | `custom_scheduler/LoraEasyCustomOptimizer/` | **DONE 2026-06-15 (wiring; needs local smoke-test)** — `git mv` + pyproject `where=[".","custom_scheduler"]` & `LoraEasyCustomOptimizer*` include (installs as top-level name → dotted `…came.CAME` configs still resolve). `library.env.ensure_custom_scheduler_on_path()` bootstraps the from-source case (called in `library/training/optimizers.py`, which `schedulers.py` imports; GUI `server.py` + `bench/speed` add the path inline). Verified short of the torch import: package gone from root, `find_spec` locates it + submodules (came/utils/CAWR) after bootstrap, ruff + config_io tests green. **Local `uv sync` + an optimizer-load smoke-test (e.g. CAME) still required to fully bless.** |
| `library/`, `networks/`, `train.py` | **unchanged at root** | = sd-scripts layout; preserves dotted `network_module` + LETS config compat |

> **Formerly-deferred moves (now landed 2026-06-15)**: both the embedder façade
> (`anima_lora/` → `library/api/`) and the optimizer zoo
> (`LoraEasyCustomOptimizer/` → `custom_scheduler/`) are done with back-compat
> shims / path bootstraps, verified as far as the torch-less container allows
> (imports/`find_spec`/ruff/tests). They couldn't be *runtime*-verified here and
> have catastrophic-if-wrong blast radius (all optimizer resolution / the public
> embedder API), so a local `uv sync` + a one-step training + optimizer-load
> smoke-test is still the gate before relying on them.

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
