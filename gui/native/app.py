# -*- coding: utf-8 -*-
"""PySide6 desktop UI for the Anima LoRA trainer.

Two parent tabs — **Training** and **Utils** — over the shared, torch-free
:mod:`gui.backend`, so this panel emits the same ``train.py`` commands as the
Gradio one; only the UI differs (native dialogs, real tables, no localhost).

Training child tabs (curated fields + schema args routed in by keyword):
- **Folder**: every path/folder picker; sample / validation / save / logging /
  resume args land here.
- **Subset**: the subset table (→ ``form['subsets']``) + global caption/shuffle
  flags.
- **Network**: adapter selection — method (LoRA type), network module/dim/alpha/
  args, LyCORIS preset + algo.
- **Optimizer**: the training-settings mega-tab — optimizer/scheduler (+args),
  loss/SNR/prior, the LR family + train-scope, norms/dropout, noise, and the
  core/hardware knobs (epochs/steps/batch/precision/swap/compile/seed) +
  flow-matching/timestep params.
- **Monitoring**: web-monitor flags.
- **Metadata**: metadata_* + no_metadata.
- **Extra**: everything uncaught (inference stacks: dcw/spectrum/spd/… ) + a raw
  ``extra_flags`` box.

Utils child tabs drive the backend util launchers: Preprocess (resize → VAE/TE/
PE/pooled caches), Update (git pull + uv sync), Auto-batch search, Masking (SAM3 +
MIT). Right panel: command preview, Start/Stop, live log, config TOML load/save.

Schema args come from ``backend.list_arg_groups()`` (needs torch to populate);
without it the curated fields still render and the structure is intact.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui import backend
from gui.modules.config_io import load_toml_to_form, save_form_to_toml

# --------------------------------------------------------------------------- #
# Curated layout — (tab, [(group_title, [(dest, label, kind), …]), …]).
# kind ∈ text | combo:<src> | tristate | bool | file | dir | scope | opthelp.
# Schema args (list_arg_groups) are routed in on top via _ROUTE_RULES; any dest
# placed here is excluded from that routing (no double-render).
# --------------------------------------------------------------------------- #
_TRAINING_TABS: list[tuple[str, list[tuple[str, list[tuple[str, str, str]]]]]] = [
    (
        "Folder",
        [
            (
                "Model paths",
                [
                    ("dit_path", "DiT checkpoint", "file"),
                    ("te_path", "Text encoder (Qwen3)", "file"),
                    ("vae_path", "VAE", "file"),
                    ("t5_tokenizer_path", "Tokenizer path", "file"),
                ],
            ),
            (
                "Output / resume / logs",
                [
                    ("output_name", "Output name", "text"),
                    ("output_dir", "Output dir", "dir"),
                    ("resume", "Resume (state dir)", "dir"),
                    ("logging_dir", "Logging dir", "dir"),
                    ("network_weights", "Warm-start weights", "file"),
                ],
            ),
            (
                "Dataset / samples",
                [
                    ("dataset_config", "Dataset config TOML", "file"),
                    ("sample_prompts", "Sample prompts file", "file"),
                ],
            ),
        ],
    ),
    ("Subset", []),
    (
        "Network",
        [
            (
                "Adapter",
                [
                    ("method", "LoRA type (method)", "combo:methods"),
                    ("network_module", "Network module", "combo:network_modules"),
                    ("network_dim", "Network dim (rank)", "text"),
                    ("network_alpha", "Network alpha", "text"),
                    ("network_args", "network_args (k=v …)", "text"),
                ],
            ),
            (
                "LyCORIS",
                [
                    ("lycoris_preset", "LyCORIS preset", "combo:lycoris_presets"),
                    ("algo", "LyCORIS algo (loha/lokr/…)", "combo:lycoris_algos"),
                ],
            ),
        ],
    ),
    (
        "Optimizer",
        [
            (
                "Optimizer",
                [
                    ("optimizer_type", "Optimizer", "combo:optimizers"),
                    ("optimizer_args", "optimizer_args (k=v …)", "text"),
                    ("optimizer_args", "↳ args help", "opthelp"),
                ],
            ),
            (
                "Scheduler",
                [
                    ("lr_scheduler", "LR scheduler (builtin)", "text"),
                    (
                        "lr_scheduler_type",
                        "LR scheduler (custom dotted path)",
                        "combo:schedulers",
                    ),
                    ("lr_scheduler_args", "lr_scheduler_args", "text"),
                    ("lr_warmup_steps", "Warmup steps", "text"),
                ],
            ),
            (
                "Learning rates / scope",
                [
                    ("learning_rate", "Learning rate", "text"),
                    ("unet_lr", "UNet / DiT LR", "text"),
                    ("text_encoder_lr", "Text-encoder LR", "text"),
                    ("llm_adapter_lr", "LLM-adapter LR", "text"),
                    ("__scope__", "Train scope", "scope"),
                ],
            ),
            (
                "Loss / regularization",
                [
                    ("loss_type", "Loss type", "text"),
                    ("network_dropout", "Network dropout", "text"),
                    ("scale_weight_norms", "Scale weight norms", "text"),
                    ("max_grad_norm", "Max grad norm", "text"),
                ],
            ),
            (
                "Core / hardware",
                [
                    ("preset", "Hardware preset", "combo:presets"),
                    ("max_train_epochs", "Max epochs", "text"),
                    ("max_train_steps", "Max steps", "text"),
                    ("train_batch_size", "Batch size", "text"),
                    ("gradient_accumulation_steps", "Grad accumulation", "text"),
                    ("blocks_to_swap", "Blocks to swap", "text"),
                    ("seed", "Seed", "text"),
                    ("mixed_precision", "Mixed precision", "combo:bf16,fp16,no"),
                    ("torch_compile", "torch.compile", "tristate"),
                ],
            ),
        ],
    ),
    (
        "Monitoring",
        [
            (
                "Web monitor",
                [
                    ("monitor", "Enable (--monitor)", "bool"),
                    ("monitor_host", "Host", "text"),
                    ("monitor_port", "Port", "text"),
                ],
            ),
        ],
    ),
    ("Metadata", []),
    ("Extra", []),
]

# Schema-arg → tab routing (ordered; first include-match wins, exclude vetoes).
# Mirrors the user's spec: Folder = paths/sample/valid/save/log; Optimizer = the
# training mega-tab; Network ≈ curated only; Metadata/Monitoring narrow; rest →
# Extra. Adjust the keyword lists to re-group.
_ROUTE_RULES: list[tuple[str, list[str], list[str]]] = [
    (
        "Folder",
        [
            "_dir",
            "_path",
            "sample",
            "valid",
            "cmmd",
            "save",
            "output",
            "huggingface",
            "hub_",
            "resume",
            "logging",
            "log_tracker",
            "console_log",
            "log_with",
            "log_prefix",
            "log_every",
            "in_json",
            "wandb",
        ],
        ["logit", "sample_ratio"],
    ),
    ("Monitoring", ["monitor"], []),
    ("Metadata", ["metadata"], []),
    (
        "Subset",
        [
            "caption",
            "shuffle",
            "weighted_caption",
            "token_warmup",
            "secondary_separator",
            "keep_tokens_separator",
            "wildcard",
        ],
        [],
    ),
    (
        "Optimizer",
        [
            "optimizer",
            "scheduler",
            "lr_",
            "_lr",
            "loss",
            "huber",
            "snr",
            "prior",
            "noise",
            "warmup",
            "decay",
            "debiased",
            "grad_norm",
            "scale_weight",
            "dropout",
            "unet_only",
            "text_encoder_only",
            "train_text_encoder",
            "timestep",
            "sigmoid",
            "weighting",
            "logit",
            "t_min",
            "t_max",
            "discrete_flow",
            "mode_scale",
            "qwen3_max_token",
            "batch",
            "blocks_to_swap",
            "block_swap",
            "checkpointing",
            "compile",
            "dynamo",
            "cudagraph",
            "mixed_precision",
            "full_bf16",
            "full_fp16",
            "fp8",
            "seed",
            "dataloader",
            "pin_memory",
            "prefetch",
            "num_workers",
            "cache",
            "accumulation",
            "max_train",
            "initial_",
            "lowram",
            "highvram",
            "offload",
            "fused",
            "activation_memory",
            "persistent",
            "unsloth",
            "channel_scal",
            # speed / VRAM knobs → sit next to compile (per user)
            "no_half_vae",
            "attn_mode",
            "attn_softmax",
            "split_attn",
            "sdpa",
            "sageattn",
            "qwen_image_vae",
            "vae_chunk",
            "vae_disable_cache",
            "text_encoder_cpu",
            # constant→cosine one-shot → sits next to the scheduler (per user)
            "constantcosine",
        ],
        ["caption", "sample_decode"],
    ),
    ("Network", ["network", "lycoris", "conv_dim", "conv_alpha"], []),
]


def _route_tab(dest: str) -> str:
    for tab, inc, exc in _ROUTE_RULES:
        if any(k in dest for k in inc) and not any(k in dest for k in exc):
            return tab
    return "Extra"


# Subset table columns → keys consumed by backend._dataset_subsets.
_SUBSET_COLS = [
    ("image_dir", "image_dir"),
    ("cache_dir", "cache_dir"),
    ("num_repeats", "num_repeats"),
    ("keep_tokens", "keep_tokens"),
    ("caption_extension", "caption_ext"),
    ("caption_dropout_rate", "cap_dropout"),
    ("batch_size", "batch_size"),
    ("tiers", "tiers (e.g. 512,1024)"),
    ("flip_aug", "flip_aug"),
    ("random_crop", "random_crop"),
]
_SUBSET_BOOL_COLS = {"flip_aug", "random_crop"}
# Flags handled by the curated train-scope combo (kept out of schema routing).
_SCOPE_FLAGS = {"network_train_unet_only", "network_train_text_encoder_only"}


def _truthy(v: object) -> bool:
    return bool(v) and str(v).lower() not in ("false", "0", "")


# Conflict / dependency greying — ported from the Gradio panel's _interactive_states.
# (target dest, predicate(driver values) → enabled). A greyed target is disabled AND
# excluded from the launch command (its value defers to the config chain).
_GREY_RULES: list[tuple[str, object]] = [
    ("huber_c", lambda v: v.get("loss_type") in ("huber", "smooth_l1")),
    ("huber_schedule", lambda v: v.get("loss_type") in ("huber", "smooth_l1")),
    ("sigmoid_scale", lambda v: v.get("timestep_sampling") in ("", "sigmoid")),
    ("logit_mean", lambda v: v.get("weighting_scheme") == "logit_normal"),
    ("logit_std", lambda v: v.get("weighting_scheme") == "logit_normal"),
    ("lr_scheduler_type", lambda v: not _truthy(v.get("use_constantcosine"))),
]
# Driver dests whose change re-evaluates the rules above + the subset-column greying.
_GREY_DRIVERS = [
    "loss_type",
    "timestep_sampling",
    "weighting_scheme",
    "use_constantcosine",
    "use_vae_cache",
    "use_text_cache",
]
# Subset table columns greyed by a cache driver (live-encoding-only knobs are inert
# once the cache is on): (col_key, driver_dest).
_SUBSET_GREY = [
    ("random_crop", "use_vae_cache"),
    ("caption_dropout_rate", "use_text_cache"),
]


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Anima LoRA — native trainer")
        self.resize(1280, 880)
        self._options = backend.options()
        self._getters: dict[str, object] = {}
        self._setters: dict[str, object] = {}
        self._adv: list[tuple[dict, object]] = []
        self._scope: QComboBox | None = None
        self._widgets: dict[str, QWidget] = {}  # dest → editable widget (for greying)
        # Dests placed explicitly → excluded from schema routing (no double render).
        self._curated: set[str] = {"extra_flags", *_SCOPE_FLAGS}
        for _tab, groups in _TRAINING_TABS:
            for _title, fields in groups:
                for dest, _label, kind in fields:
                    if kind not in ("opthelp", "scope"):
                        self._curated.add(dest)
        # Partition schema args (arg_groups) into Training tabs by route.
        self._tab_schema: dict[str, list[dict]] = {}
        for group in self._options.get("arg_groups") or []:
            for arg in group.get("args") or []:
                d = arg.get("dest") or ""
                if d in self._curated:
                    continue
                self._tab_schema.setdefault(_route_tab(d), []).append(arg)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_parent_tabs())
        splitter.addWidget(self._build_run_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        self._wire_greying()
        self._apply_greying()

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    # ----- parent tabs ---------------------------------------------------- #
    def _build_parent_tabs(self) -> QTabWidget:
        parent = QTabWidget()
        parent.addTab(self._build_training_parent(), "Training")
        parent.addTab(self._build_utils_parent(), "Utils")
        return parent

    def _build_training_parent(self) -> QTabWidget:
        inner = QTabWidget()
        for tab_name, groups in _TRAINING_TABS:
            inner.addTab(
                self._scroll(self._build_training_tab(tab_name, groups)), tab_name
            )
        return inner

    def _build_training_tab(self, tab_name: str, groups: list) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        for title, fields in groups:
            vbox.addWidget(self._build_group(title, fields))
        if tab_name == "Subset":
            vbox.addWidget(self._build_subset_box())
        # Schema args routed into this tab (populated only when torch is present).
        schema = self._tab_schema.get(tab_name) or []
        if schema:
            box = QGroupBox("More flags")
            form = QFormLayout(box)
            for arg in sorted(schema, key=lambda a: a.get("dest") or ""):
                form.addRow(
                    arg.get("dest") or arg.get("flag"), self._build_adv_field(arg)
                )
            vbox.addWidget(box)
        if tab_name == "Extra":
            vbox.addWidget(self._build_extra_flags_box())
        vbox.addStretch(1)
        return w

    def _scroll(self, inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    # ----- curated field widgets ------------------------------------------ #
    def _build_group(self, title: str, fields: list[tuple[str, str, str]]) -> QGroupBox:
        gb = QGroupBox(title)
        form = QFormLayout(gb)
        for dest, label, kind in fields:
            form.addRow(label, self._build_field(dest, kind))
        return gb

    def _build_field(self, dest: str, kind: str) -> QWidget:
        if kind == "scope":
            combo = QComboBox()
            combo.addItems(["both (UNet + TE)", "UNet only", "TE only"])
            self._scope = combo
            return combo
        if kind == "opthelp":
            btn = QPushButton("show")
            btn.clicked.connect(self._show_optimizer_help)
            return btn
        if kind == "bool":
            cb = QCheckBox()
            self._getters[dest] = lambda c=cb: c.isChecked()
            self._setters[dest] = lambda v, c=cb: c.setChecked(_truthy(v))
            self._widgets[dest] = cb
            return cb
        if kind == "tristate":
            combo = QComboBox()
            combo.addItems(["", "on", "off"])
            self._getters[dest] = lambda c=combo: c.currentText().strip()
            self._setters[dest] = lambda v, c=combo: c.setCurrentText(str(v or ""))
            self._widgets[dest] = combo
            return combo
        if kind.startswith("combo:"):
            src = kind.split(":", 1)[1]
            items = self._options.get(src) if src in self._options else src.split(",")
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItem("")
            combo.addItems([str(x) for x in (items or [])])
            self._getters[dest] = lambda c=combo: c.currentText().strip()
            self._setters[dest] = lambda v, c=combo: c.setCurrentText(str(v or ""))
            self._widgets[dest] = combo
            return combo
        edit = QLineEdit()
        self._getters[dest] = lambda e=edit: e.text().strip()
        self._setters[dest] = lambda v, e=edit: e.setText(str(v or ""))
        self._widgets[dest] = edit
        if kind in ("file", "dir"):
            row = QWidget()
            hb = QHBoxLayout(row)
            hb.setContentsMargins(0, 0, 0, 0)
            hb.addWidget(edit)
            btn = QPushButton("📁")
            btn.setFixedWidth(36)
            btn.clicked.connect(lambda _=False, e=edit, k=kind: self._browse(e, k))
            hb.addWidget(btn)
            return row
        return edit

    def _browse(self, edit: QLineEdit, kind: str) -> None:
        start = edit.text().strip() or str(backend.ROOT)
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", start)
        if path:
            edit.setText(path)

    def _show_optimizer_help(self) -> None:
        name = ""
        getter = self._getters.get("optimizer_type")
        if getter:
            name = str(getter() or "")
        if not name:
            QMessageBox.information(self, "Optimizer args", "Pick an optimizer first.")
            return
        try:
            info = backend.optimizer_arg_help(name)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Optimizer args", str(exc))
            return
        if not info:
            QMessageBox.information(
                self, "Optimizer args", f"No arg help for {name!r}."
            )
            return
        lines = [f"• {k}: {v}" for k, v in info.items()]
        QMessageBox.information(self, f"{name} — optimizer_args", "\n".join(lines))

    # ----- schema (auto) field widgets ------------------------------------ #
    def _build_adv_field(self, arg: dict) -> QWidget:
        flag = arg.get("flag")
        help_txt = arg.get("help") or ""
        if arg.get("negatable"):
            combo = QComboBox()
            combo.addItems(["default", "on", "off"])
            combo.setToolTip(help_txt)
            self._adv.append(
                (
                    arg,
                    lambda c=combo, f=flag: (
                        {"flag": f, "negatable": True, "tri": c.currentText()}
                        if c.currentText() != "default"
                        else None
                    ),
                )
            )
            self._widgets[arg.get("dest") or ""] = combo
            return combo
        if arg.get("is_bool"):
            cb = QCheckBox()
            cb.setToolTip(help_txt)
            self._adv.append(
                (
                    arg,
                    lambda c=cb, f=flag: (
                        {"flag": f, "is_bool": True, "value": True, "on": True}
                        if c.isChecked()
                        else None
                    ),
                )
            )
            self._widgets[arg.get("dest") or ""] = cb
            return cb
        if arg.get("choices"):
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItem("")
            combo.addItems([str(x) for x in arg["choices"]])
            combo.setToolTip(help_txt)
            self._adv.append(
                (
                    arg,
                    lambda c=combo, f=flag, a=arg: (
                        {
                            "flag": f,
                            "value": c.currentText().strip(),
                            "nargs": a.get("nargs"),
                            "on": True,
                        }
                        if c.currentText().strip()
                        else None
                    ),
                )
            )
            self._widgets[arg.get("dest") or ""] = combo
            return combo
        edit = QLineEdit()
        edit.setToolTip(help_txt)
        self._adv.append(
            (
                arg,
                lambda e=edit, f=flag, a=arg: (
                    {
                        "flag": f,
                        "value": e.text().strip(),
                        "nargs": a.get("nargs"),
                        "on": True,
                    }
                    if e.text().strip()
                    else None
                ),
            )
        )
        self._widgets[arg.get("dest") or ""] = edit
        return edit

    # ----- greying (conflict / dependency) -------------------------------- #
    def _widget_value(self, dest: str) -> object:
        w = self._widgets.get(dest)
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText().strip()
        if isinstance(w, QLineEdit):
            return w.text().strip()
        return None

    def _wire_greying(self) -> None:
        for dest in _GREY_DRIVERS:
            w = self._widgets.get(dest)
            if isinstance(w, QComboBox):
                w.currentTextChanged.connect(lambda *_: self._apply_greying())
            elif isinstance(w, QLineEdit):
                w.textChanged.connect(lambda *_: self._apply_greying())
            elif isinstance(w, QCheckBox):
                w.toggled.connect(lambda *_: self._apply_greying())

    def _apply_greying(self) -> None:
        vals = {d: self._widget_value(d) for d in _GREY_DRIVERS}
        for target, pred in _GREY_RULES:
            w = self._widgets.get(target)
            if w is not None:
                w.setEnabled(bool(pred(vals)))
        cols = {k: i for i, (k, _) in enumerate(_SUBSET_COLS)}
        for col_key, driver in _SUBSET_GREY:
            enabled = not _truthy(vals.get(driver))
            ci = cols[col_key]
            for r in range(self._subset_table.rowCount()):
                self._set_cell_enabled(r, ci, enabled, col_key in _SUBSET_BOOL_COLS)

    def _set_cell_enabled(
        self, row: int, col: int, enabled: bool, checkable: bool
    ) -> None:
        item = self._subset_table.item(row, col)
        if item is None:
            return
        flags = Qt.ItemIsSelectable
        if enabled:
            flags |= Qt.ItemIsEnabled | (
                Qt.ItemIsUserCheckable if checkable else Qt.ItemIsEditable
            )
        item.setFlags(flags)

    def _build_extra_flags_box(self) -> QGroupBox:
        gb = QGroupBox("Raw extra flags")
        form = QFormLayout(gb)
        edit = QPlainTextEdit()
        edit.setMaximumHeight(70)
        edit.setPlaceholderText("--highvram\n--guidance_scale 1.0")
        self._getters["extra_flags"] = lambda e=edit: e.toPlainText().strip()
        self._setters["extra_flags"] = lambda v, e=edit: e.setPlainText(str(v or ""))
        form.addRow("Anything else", edit)
        return gb

    # ----- subset table --------------------------------------------------- #
    def _build_subset_box(self) -> QGroupBox:
        gb = QGroupBox(
            "Subsets (each row → one [[datasets.subsets]]; empty = fallback above)"
        )
        vbox = QVBoxLayout(gb)
        self._subset_table = QTableWidget(0, len(_SUBSET_COLS))
        self._subset_table.setHorizontalHeaderLabels([h for _, h in _SUBSET_COLS])
        self._subset_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._subset_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        vbox.addWidget(self._subset_table)
        btns = QHBoxLayout()
        add_folder = QPushButton("➕ Add subset (folder…)")
        add_row = QPushButton("➕ Add empty row")
        rm_row = QPushButton("➖ Remove selected")
        add_folder.clicked.connect(self._subset_add_folder)
        add_row.clicked.connect(lambda: self._subset_add_row())
        rm_row.clicked.connect(self._subset_remove)
        for b in (add_folder, add_row, rm_row):
            btns.addWidget(b)
        btns.addStretch(1)
        vbox.addLayout(btns)
        return gb

    def _subset_add_row(self, values: dict | None = None) -> None:
        values = values or {}
        r = self._subset_table.rowCount()
        self._subset_table.insertRow(r)
        for c, (key, _) in enumerate(_SUBSET_COLS):
            if key in _SUBSET_BOOL_COLS:
                item = QTableWidgetItem()
                item.setFlags(
                    Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
                )
                on = bool(values.get(key)) and str(values.get(key)).lower() not in (
                    "false",
                    "0",
                    "",
                )
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            else:
                item = QTableWidgetItem(str(values.get(key, "") or ""))
            self._subset_table.setItem(r, c, item)
        if hasattr(self, "_widgets"):
            self._apply_greying()  # grey the new row's cache-gated cells

    def _subset_add_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select image folder", str(backend.ROOT)
        )
        if path:
            self._subset_add_row({"image_dir": path})

    def _subset_remove(self) -> None:
        rows = sorted(
            {i.row() for i in self._subset_table.selectedIndexes()}, reverse=True
        )
        for r in rows:
            self._subset_table.removeRow(r)

    def _collect_subsets(self) -> list[dict]:
        out: list[dict] = []
        for r in range(self._subset_table.rowCount()):
            row: dict = {}
            for c, (key, _) in enumerate(_SUBSET_COLS):
                item = self._subset_table.item(r, c)
                if item is None or not (item.flags() & Qt.ItemIsEnabled):
                    continue  # greyed cell (cache-gated) → inert
                if key in _SUBSET_BOOL_COLS:
                    if item.checkState() == Qt.Checked:
                        row[key] = True
                elif item.text().strip():
                    row[key] = item.text().strip()
            if row.get("image_dir"):
                out.append(row)
        return out

    # ----- utils parent --------------------------------------------------- #
    def _build_utils_parent(self) -> QTabWidget:
        inner = QTabWidget()
        inner.addTab(self._scroll(self._build_preprocess_tab()), "Preprocess")
        inner.addTab(self._scroll(self._build_update_tab()), "Update")
        inner.addTab(self._scroll(self._build_autobatch_tab()), "Auto-batch")
        inner.addTab(self._scroll(self._build_masking_tab()), "Masking")
        return inner

    def _build_preprocess_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "Resize → cache (VAE / TE / PE / pooled). Reads configs/preprocess.toml "
                "+ base.toml for paths/target_res. Mutually exclusive with a run; output "
                "streams to the log."
            )
        )
        steps = [
            ("All (resize → cache)", "all"),
            ("Resize", "resize"),
            ("VAE latents", "vae"),
            ("Text-encoder", "te"),
            ("PE features", "pe"),
            ("Pooled TE", "pooled"),
            ("Reconcile (drop stale)", "reconcile"),
        ]
        for label, step in steps:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, s=step: self._do_preprocess(s))
            vbox.addWidget(btn)
        vbox.addStretch(1)
        return w

    def _do_preprocess(self, step: str) -> None:
        res = backend.run_preprocess(step)
        if not res.get("ok"):
            QMessageBox.warning(self, "Preprocess", str(res.get("error") or res))

    def _build_update_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "Update the tool (git pull + uv sync) — datasets/models are gitignored."
            )
        )
        self._update_info = QLabel("—")
        self._update_info.setWordWrap(True)
        vbox.addWidget(self._update_info)
        row = QHBoxLayout()
        check = QPushButton("Check for updates")
        do = QPushButton("Update now")
        check.clicked.connect(self._do_check_update)
        do.clicked.connect(self._do_update)
        row.addWidget(check)
        row.addWidget(do)
        row.addStretch(1)
        vbox.addLayout(row)
        vbox.addStretch(1)
        return w

    def _do_check_update(self) -> None:
        try:
            v = backend.tool_version(fetch=True)
        except Exception as exc:  # noqa: BLE001
            self._update_info.setText(f"error: {exc}")
            return
        self._update_info.setText(
            f"branch {v.get('branch')} · sha {v.get('sha')} · ahead {v.get('ahead')} "
            f"behind {v.get('behind')} · {'up to date' if v.get('up_to_date') else 'update available'}"
            + (f"\n{v.get('note')}" if v.get("note") else "")
        )

    def _do_update(self) -> None:
        try:
            res = backend.update_tool()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Update", str(exc))
            return
        QMessageBox.information(self, "Update", str(res.get("note") or res))

    def _build_autobatch_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "Max-batch search (tasks.py bench-autobatch) using the current Training "
                "fields. Mutually exclusive with a run; output streams to the log."
            )
        )
        btn = QPushButton("Run auto-batch search")
        btn.clicked.connect(self._do_autobatch)
        vbox.addWidget(btn)
        vbox.addStretch(1)
        return w

    def _do_autobatch(self) -> None:
        res = backend.bench_autobatch(self._collect())
        if not res.get("ok"):
            QMessageBox.warning(self, "Auto-batch", str(res.get("error") or res))

    def _build_masking_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        gb = QGroupBox("Masking (SAM3 + MIT → merged masks)")
        form = QFormLayout(gb)
        self._mask_sam = QCheckBox()
        self._mask_sam.setChecked(True)
        self._mask_mit = QCheckBox()
        self._mask_mit.setChecked(True)
        self._mit_tt = QLineEdit()
        self._mit_dilate = QLineEdit()
        form.addRow("SAM3", self._mask_sam)
        form.addRow("MIT (text removal)", self._mask_mit)
        form.addRow("MIT text threshold", self._mit_tt)
        form.addRow("MIT dilate", self._mit_dilate)
        vbox.addWidget(gb)
        btn = QPushButton("Run masking")
        btn.clicked.connect(self._do_masking)
        vbox.addWidget(btn)
        vbox.addStretch(1)
        return w

    def _do_masking(self) -> None:
        form = {
            "mask_sam": self._mask_sam.isChecked(),
            "mask_mit": self._mask_mit.isChecked(),
            "mit_text_threshold": self._mit_tt.text().strip(),
            "mit_dilate": self._mit_dilate.text().strip(),
        }
        res = backend.run_masking(form)
        if not res.get("ok"):
            QMessageBox.warning(self, "Masking", str(res.get("error") or res))

    # ----- run panel ------------------------------------------------------ #
    def _build_run_panel(self) -> QWidget:
        panel = QWidget()
        vbox = QVBoxLayout(panel)
        cfg_row = QHBoxLayout()
        btn_load = QPushButton("Load config…")
        btn_save = QPushButton("Save config…")
        btn_load.clicked.connect(self._load_config)
        btn_save.clicked.connect(self._save_config)
        cfg_row.addWidget(btn_load)
        cfg_row.addWidget(btn_save)
        cfg_row.addStretch(1)
        vbox.addLayout(cfg_row)

        vbox.addWidget(QLabel("Command preview"))
        self._preview = QPlainTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setMaximumHeight(120)
        self._preview.setFont(QFont("monospace"))
        vbox.addWidget(self._preview)

        btn_row = QHBoxLayout()
        self._btn_preview = QPushButton("Preview")
        self._btn_start = QPushButton("▶ Start")
        self._btn_stop = QPushButton("■ Stop")
        self._btn_monitor = QPushButton("Open monitor")
        self._btn_preview.clicked.connect(self._do_preview)
        self._btn_start.clicked.connect(self._do_start)
        self._btn_stop.clicked.connect(self._do_stop)
        self._btn_monitor.clicked.connect(self._open_monitor)
        self._btn_monitor.setEnabled(False)
        for b in (
            self._btn_preview,
            self._btn_start,
            self._btn_stop,
            self._btn_monitor,
        ):
            btn_row.addWidget(b)
        vbox.addLayout(btn_row)

        self._status = QLabel("idle")
        vbox.addWidget(self._status)
        vbox.addWidget(QLabel("Log"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("monospace"))
        vbox.addWidget(self._log, 1)
        return panel

    # ----- form <-> dict -------------------------------------------------- #
    def _enabled(self, dest: str) -> bool:
        # A greyed (disabled) field is inert: excluded from the command so its
        # value defers to the config chain (matches the Gradio panel).
        w = self._widgets.get(dest)
        return w is None or w.isEnabled()

    def _collect(self) -> dict:
        form = {
            dest: get() for dest, get in self._getters.items() if self._enabled(dest)
        }
        subsets = self._collect_subsets()
        if subsets:
            form["subsets"] = subsets
        adv = [
            item
            for a, g in self._adv
            if self._enabled(a.get("dest") or "") and (item := g())
        ]
        if self._scope is not None:
            idx = self._scope.currentIndex()
            if idx == 1:
                adv.append(
                    {
                        "flag": "--network_train_unet_only",
                        "is_bool": True,
                        "value": True,
                        "on": True,
                    }
                )
            elif idx == 2:
                adv.append(
                    {
                        "flag": "--network_train_text_encoder_only",
                        "is_bool": True,
                        "value": True,
                        "on": True,
                    }
                )
        if adv:
            form["adv"] = adv
        return form

    def _apply(self, form: dict) -> None:
        for dest, val in form.items():
            setter = self._setters.get(dest)
            if setter:
                setter(val)
        subsets = form.get("subsets")
        if isinstance(subsets, list):
            self._subset_table.setRowCount(0)
            for s in subsets:
                if isinstance(s, dict):
                    self._subset_add_row(s)
        self._apply_greying()  # re-evaluate after a config load changes drivers

    # ----- actions -------------------------------------------------------- #
    def _do_preview(self) -> None:
        try:
            self._preview.setPlainText(" ".join(backend.build_command(self._collect())))
        except Exception as exc:  # noqa: BLE001
            self._preview.setPlainText(f"[preview error] {exc}")

    def _do_start(self) -> None:
        self._do_preview()
        res = backend.launch(self._collect())
        if not res.get("ok"):
            QMessageBox.critical(self, "Launch failed", str(res.get("error") or res))
            return
        self._log.clear()

    def _do_stop(self) -> None:
        res = backend.stop()
        if not res.get("ok"):
            QMessageBox.warning(self, "Stop", str(res.get("error") or res))

    def _open_monitor(self) -> None:
        url = backend.status().get("monitor_url")
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config TOML", str(backend.ROOT), "TOML (*.toml);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                form = load_toml_to_form(f.read())
            self._apply(form)
            self._do_preview()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(exc))

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config TOML", str(backend.ROOT / "config.toml"), "TOML (*.toml)"
        )
        if not path:
            return
        try:
            text = save_form_to_toml(self._collect())
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))

    # ----- polling -------------------------------------------------------- #
    def _poll(self) -> None:
        st = backend.status()
        running = bool(st.get("running"))
        self._btn_start.setEnabled(not running)
        self._btn_stop.setEnabled(running)
        self._btn_monitor.setEnabled(bool(st.get("monitor_url")))
        if running:
            self._status.setText(
                f"running · pid {st.get('pid')} · {int(st.get('elapsed') or 0)}s"
            )
        elif st.get("returncode") is not None:
            self._status.setText(f"finished · exit {st.get('returncode')}")
        else:
            self._status.setText("idle")
        lines = backend.log_tail(400).get("lines") or []
        if lines:
            sb = self._log.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 4
            self._log.setPlainText("\n".join(lines))
            if at_bottom:
                sb.setValue(sb.maximum())


def run() -> None:
    """Create the QApplication and show the main window (blocking)."""
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    run()
