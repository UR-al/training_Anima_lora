# -*- coding: utf-8 -*-
"""PySide6 desktop UI for the Anima LoRA trainer.

Tabbed form (Training / Dataset / Advanced) over the shared, torch-free
:mod:`gui.backend`, so this panel emits the same ``train.py`` commands as the
Gradio one — only the UI differs (native dialogs, real tables, no localhost).

- **Training**: curated fields built from a small declarative spec, dropdowns
  sourced from ``backend.options()``.
- **Dataset**: a real subset table → ``form['subsets']`` (the backend normalizes
  it into a precached ``--dataset_config``), plus a single-folder fallback.
- **Advanced**: every other ``train.py`` flag, schema-driven from
  ``backend.list_arg_groups()`` → ``form['adv']`` (per-flag widgets with help),
  plus a raw ``extra_flags`` escape hatch.

Command preview / Start / Stop / live log + config TOML load/save (config_io)
live in the right-hand run panel. Utils + saved-run queue are later milestones.
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

# Curated field spec: (dest, label, kind). kind ∈ text | combo:<src> | tristate |
# bool | file | dir. combo src is an options() key or a literal comma list.
_TRAINING_SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
    (
        "Model files",
        [
            ("dit_path", "DiT checkpoint", "file"),
            ("te_path", "Text encoder (Qwen3)", "file"),
            ("vae_path", "VAE", "file"),
        ],
    ),
    (
        "Output",
        [
            ("output_name", "Output name", "text"),
            ("output_dir", "Output dir", "dir"),
        ],
    ),
    (
        "Method / Network",
        [
            ("method", "Method", "combo:methods"),
            ("preset", "Preset", "combo:presets"),
            ("network_module", "Network module", "combo:network_modules"),
            ("network_dim", "Network dim (rank)", "text"),
            ("network_alpha", "Network alpha", "text"),
            ("network_weights", "Warm-start weights", "file"),
            ("network_args", "network_args (k=v …)", "text"),
        ],
    ),
    (
        "Optimizer / Schedule",
        [
            ("optimizer_type", "Optimizer", "combo:optimizers"),
            ("learning_rate", "Learning rate", "text"),
            ("unet_lr", "DiT / unet LR", "text"),
            ("optimizer_args", "optimizer_args (k=v …)", "text"),
            ("lr_scheduler_type", "LR scheduler (custom)", "combo:schedulers"),
            ("lr_scheduler", "LR scheduler (builtin)", "text"),
            ("lr_scheduler_args", "lr_scheduler_args", "text"),
            ("lr_warmup_steps", "Warmup steps", "text"),
        ],
    ),
    (
        "Training",
        [
            ("max_train_epochs", "Max epochs", "text"),
            ("max_train_steps", "Max steps", "text"),
            ("train_batch_size", "Batch size", "text"),
            ("gradient_accumulation_steps", "Grad accumulation", "text"),
            ("blocks_to_swap", "Blocks to swap", "text"),
            ("mixed_precision", "Mixed precision", "combo:bf16,fp16,no"),
            ("seed", "Seed", "text"),
            ("torch_compile", "torch.compile", "tristate"),
            ("resume", "Resume (state dir)", "dir"),
        ],
    ),
    (
        "Samples / Monitor",
        [
            ("sample_prompts", "Sample prompts file", "file"),
            ("sample_every_n_epochs", "Sample every N epochs", "text"),
            ("monitor", "Web monitor (--monitor)", "bool"),
            ("monitor_port", "Monitor port", "text"),
        ],
    ),
]
_DATASET_SECTION = (
    "Dataset (single-folder fallback)",
    [
        ("dataset_config", "Dataset config TOML", "file"),
        ("raw_image_dir", "…or a single image folder", "dir"),
    ],
)

# Subset table columns → keys consumed by backend._dataset_subsets. Strings are
# fine (the backend casts num_repeats/keep_tokens/… itself); the two aug columns
# are checkboxes.
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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Anima LoRA — native trainer")
        self.resize(1180, 840)
        self._options = backend.options()
        self._getters: dict[str, object] = {}
        self._setters: dict[str, object] = {}
        self._adv: list[tuple[dict, object]] = []

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_tabs())
        splitter.addWidget(self._build_run_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    # ----- tabs ----------------------------------------------------------- #
    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._scroll(self._sections_widget(_TRAINING_SECTIONS)), "Training")
        tabs.addTab(self._scroll(self._build_dataset_tab()), "Dataset")
        tabs.addTab(self._scroll(self._build_advanced_tab()), "Advanced")
        return tabs

    def _scroll(self, inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

    def _sections_widget(self, sections) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        for title, fields in sections:
            vbox.addWidget(self._build_group(title, fields))
        vbox.addStretch(1)
        return w

    def _build_group(self, title: str, fields: list[tuple[str, str, str]]) -> QGroupBox:
        gb = QGroupBox(title)
        form = QFormLayout(gb)
        for dest, label, kind in fields:
            form.addRow(label, self._build_field(dest, kind))
        return gb

    def _build_field(self, dest: str, kind: str) -> QWidget:
        if kind == "bool":
            cb = QCheckBox()
            self._getters[dest] = lambda c=cb: c.isChecked()
            self._setters[dest] = lambda v, c=cb: c.setChecked(
                bool(v) and str(v).lower() not in ("false", "0", "")
            )
            return cb
        if kind == "tristate":
            combo = QComboBox()
            combo.addItems(["", "on", "off"])
            self._getters[dest] = lambda c=combo: c.currentText().strip()
            self._setters[dest] = lambda v, c=combo: c.setCurrentText(str(v or ""))
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
            return combo
        # text / file / dir → a line edit, with a Browse button for paths.
        edit = QLineEdit()
        self._getters[dest] = lambda e=edit: e.text().strip()
        self._setters[dest] = lambda v, e=edit: e.setText(str(v or ""))
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

    # ----- dataset tab (subset table) ------------------------------------- #
    def _build_dataset_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)
        vbox.addWidget(
            QLabel(
                "Subsets drive training (each row → one [[datasets.subsets]]). Leave the "
                "table empty to use the single-folder fallback below."
            )
        )
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

        vbox.addWidget(self._build_group(*_DATASET_SECTION))
        vbox.addStretch(1)
        return w

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
                checked = bool(values.get(key)) and str(
                    values.get(key)
                ).lower() not in ("false", "0", "")
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            else:
                item = QTableWidgetItem(str(values.get(key, "") or ""))
            self._subset_table.setItem(r, c, item)

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
                if item is None:
                    continue
                if key in _SUBSET_BOOL_COLS:
                    if item.checkState() == Qt.Checked:
                        row[key] = True
                else:
                    val = item.text().strip()
                    if val:
                        row[key] = val
            if row.get("image_dir"):
                out.append(row)
        return out

    # ----- advanced tab (schema-driven, all flags) ------------------------ #
    def _build_advanced_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)

        gb = QGroupBox("Raw extra flags")
        form = QFormLayout(gb)
        edit = QPlainTextEdit()
        edit.setMaximumHeight(60)
        edit.setPlaceholderText("--highvram\n--guidance_scale 1.0")
        self._getters["extra_flags"] = lambda e=edit: e.toPlainText().strip()
        self._setters["extra_flags"] = lambda v, e=edit: e.setPlainText(str(v or ""))
        form.addRow("Anything else", edit)
        vbox.addWidget(gb)

        for group in self._options.get("arg_groups") or []:
            box = QGroupBox(str(group.get("role") or "args"))
            gform = QFormLayout(box)
            for arg in group.get("args") or []:
                widget = self._build_adv_field(arg)
                gform.addRow(arg.get("dest") or arg.get("flag"), widget)
            vbox.addWidget(box)
        vbox.addStretch(1)
        return w

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
        return edit

    def _collect_adv(self) -> list[dict]:
        out = []
        for _arg, getter in self._adv:
            item = getter()
            if item:
                out.append(item)
        return out

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

        vbox.addWidget(QLabel("Training log"))
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("monospace"))
        vbox.addWidget(self._log, 1)
        return panel

    # ----- form <-> dict -------------------------------------------------- #
    def _collect(self) -> dict:
        form = {dest: get() for dest, get in self._getters.items()}
        subsets = self._collect_subsets()
        if subsets:
            form["subsets"] = subsets
        adv = self._collect_adv()
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
