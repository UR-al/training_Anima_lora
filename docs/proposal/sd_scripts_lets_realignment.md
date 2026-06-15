# sd-scripts / LETS realignment — target structure proposal

Status: **DRAFT for approval** (structure target first, then GUI on top).
Date: 2026-06-15.

Goal (from the requester): drive the trainer in plain **sd-scripts / LoRA_Easy_Training_Scripts (LETS)**
style — one `--config_file` carrying everything, `network_module` + `network_args`
carrying the routing, a kohya-style GUI on top that emits such a config, **no
daemon** (direct CLI), and a folder layout that reads cleanly instead of being
named after where each piece was lifted from.

---

## 0. Key finding — the trainer is already ~80% sd-scripts/LETS-shaped

Before renaming anything, note what already matches the convention (evidence):

| Concern | Already sd-scripts/LETS? | Evidence |
|---|---|---|
| Network loading | **Yes** | `train.py:1735` `importlib.import_module(network_module)` → `create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, **kwargs)` + `create_network_from_weights(...)`. Identical to sd-scripts. |
| Pure config-file run | **Yes** | `library/config/io.py` `--config_file` alone is **lenient** (unknown keys ignored unless `--config_strict`); routing rides `network_args`. So `network_module=… , network_args=[…]` runs with **no** method/preset. |
| Routing via `network_args` | **Yes** | three-axis `use_moe_style` / `route_per_layer` / `router_source` are already in the `NETWORK_KWARGS` allowlist and consumed from `network_args`. |
| Direct (non-daemon) launch | **Yes** | `scripts/tasks/_common.py` `build_launch_cmd()` / `accelerate_launch()` is the default inline path; the daemon is opt-in (`--queue`). |
| LETS/kohya config import | **Yes** | `scripts/webgui/server.py:2050 import_config()` already converts kohya / LoRA_Easy sectioned TOML/JSON. |
| Core package names | **Yes** | `library/` and `networks/` are the **sd-scripts** names; `LoraEasyCustomOptimizer/` is the **LETS** package's own name. |

**Implication:** "pure sd-scripts/LETS config → CLI" is largely a matter of
*blessing and documenting the path that already exists* + cleanup + the GUI —
not a ground-up rewrite. And renaming `library` / `networks` /
`LoraEasyCustomOptimizer` would move **away** from the convention, not toward it.

---

## 1. Proposed target folder layout

Principle: keep the sd-scripts/LETS-standard names; make the **vendor boundary**
explicit; give the one bespoke "lumped" name (`anima_lora`) a clear role.

```
training_Anima_lora/
├── train.py, inference.py            # root train/infer scripts        (sd-scripts ✓, keep)
├── library/                          # core trainer subsystem          (sd-scripts ✓, keep)
├── networks/                         # adapters; importlib create_network (sd-scripts ✓, keep)
├── configs/                          # config TOMLs + LETS-style examples (keep)
├── gui/                              # ← NEW home for ALL GUI code
│   ├── kohya/                        #   vendored kohya_ss LoRA + Utilities tab frame
│   └── webgui/                       #   (moved) the stdlib panel, if kept
├── vendor/                           # ← explicit third-party boundary
│   └── lora_easy_optimizer/          #   (moved) = today's LoraEasyCustomOptimizer*
├── tools/                            # sd-scripts-style standalone utilities (optional, from scripts/)
├── bench/                            # keep
└── anima_lora/  → DECISION NEEDED    # embedder façade; see §4
```

\* `LoraEasyCustomOptimizer` is imported by only ~3 files + `pyproject` + the
resolver in `library/training/optimizers.py`. A move is cheap **iff** we keep a
re-export shim so the friendly-name registry and any saved configs keep working.
Counter-argument: it is the upstream LETS name — moving it diverges from LETS.
**Recommend: keep the name, but document it as vendored** (cheapest, most
LETS-faithful). Open for decision in §4.

### What does NOT get renamed (and why)
- `library/`, `networks/` — already the sd-scripts names (518 / 104 refs). Renaming
  is pure cost with negative convention value.
- `LoraEasyCustomOptimizer/` — the actual LETS package name.

---

## 2. Config model — pure `--config_file` (LETS/sd-scripts style)

**Primary CLI entry becomes:**
```bash
python train.py --config_file path/to/run.toml
# (or: accelerate launch train.py --config_file …  for multi-GPU)
```
- The config carries **everything**: `network_module`, `network_args` (incl.
  routing), optimizer/scheduler, dataset blueprint, sample settings, `--monitor`.
- `--method`/`--preset` stay as an **optional convenience layer** (unchanged for
  existing users), but are **not** required and are **not combined** with
  `--config_file` (the `io.py:721` branch already enforces "config_file wins").
- We ship **LETS-style example configs** under `configs/examples/` (a
  `lora.toml`, `lora_moe.toml`, etc.) that are pure config-file runnable.

Example (illustrative) `configs/examples/lora.toml`:
```toml
network_module = "networks.lora_anima"
network_dim    = 32
network_alpha  = 16
network_args   = ["use_moe_style=shared_A", "route_per_layer=true", "router_source=fei"]
optimizer_type = "CAME"
learning_rate  = 1e-4
max_train_epochs = 16
# + dataset blueprint ([[datasets]] / [[datasets.subsets]]) and sample settings
```

**Work needed:** mostly validation + docs + examples; the loader path already exists.
Optional: relax/standardize `import_config` into a first-class
`--config_file` LETS reader so an unmodified LETS config "just runs".

---

## 3. Network module convention

Already matches (importlib + `create_network`/`create_network_from_weights`).
Optional cosmetic alignment to the sd-scripts single-file form
(`networks/lora.py` instead of `networks/lora_anima/`) is **not recommended** —
`lora_anima/` is a justified 6-module package and 104 refs point at it. Keep the
dotted-path entry points; they are the contract LETS configs key off.

---

## 4. Open naming decisions (need confirmation)

1. **Rename the already-standard packages?** Recommendation: **No** for
   `library`/`networks`/`LoraEasyCustomOptimizer` (renaming diverges from
   sd-scripts/LETS). Only introduce `gui/` and `vendor/` (or keep optimizer at
   root). — *Requester earlier said "rename core packages too"; this finding may
   change that call.*
2. **`anima_lora/` façade** (the one bespoke "lumped" name, ~9 importers + the
   public embedder API): keep as-is / rename (e.g. `anima/` or `embed/`) / fold
   into `library/api/`. Trade-off: it is the documented programmatic front door,
   so a rename ripples into examples + ComfyUI nodes + CLAUDE.md.
3. **GUI consolidation**: move `scripts/webgui/` and the vendored kohya GUI under
   a single `gui/` tree? (Recommended.)

---

## 5. Daemon removal

- The GUI's "Start" calls the **direct** path (`build_launch_cmd`) — no
  `ensure_daemon`/`submit`. Logs go to the terminal (sd-scripts style).
- `--queue` and the daemon package can be **retained but unused** by the GUI
  (lower risk) or removed outright (ComfyUI trainer node + `--queue` flows would
  need the inline path). Recommendation: **stop using it from the GUI now**;
  decide on full removal separately (it also backs the ComfyUI trainer node).

---

## 6. Phasing (after this structure is approved)

- **Phase 1 — config + launch**: bless `--config_file` LETS path, add
  `configs/examples/*.toml`, switch GUI/launch to direct CLI (no daemon), docs.
- **Phase 2 — layout**: introduce `gui/` (+ `vendor/`); move webgui; resolve the
  `anima_lora` decision with re-export shims so imports don't break.
- **Phase 3 — GUI**: vendor kohya LoRA + Utilities tabs into `gui/kohya/`, rewire
  the Train button to emit a pure config + `python train.py --config_file …`, add
  the Anima toggles (method/preset optional, `--monitor`).
- Each phase: keep re-export shims for renamed packages; run `make test-unit`.
```
