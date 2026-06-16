# -*- coding: utf-8 -*-
"""PySide6 desktop UI for the Anima LoRA trainer (Milestone 1: core flow).

Curated training fields + an Extra-flags escape hatch (covers every other
``train.py`` flag via the backend's ``extra_flags`` passthrough) → command
preview → Start/Stop → live log tail. Config TOML load/save rides
``gui.modules.config_io`` (same round-trip as the Gradio panel). Dataset subset
editor / schema-driven Extra widgets / Utils are follow-up milestones; the
backend already supports them, so it's purely UI build-out.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import backend
from gui.modules.config_io import load_toml_to_form, save_form_to_toml

# Field spec: (dest, label, kind). kind ∈ text | combo:<src> | tristate | bool |
# file | dir | multiline. combo src is an options() key or a comma list.
_SECTIONS: list[tuple[str, list[tuple[str, str, str]]]] = [
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
        "Dataset",
        [
            ("dataset_config", "Dataset config TOML", "file"),
            ("raw_image_dir", "…or a single image folder", "dir"),
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
_EXTRA_FIELD = (
    "extra_flags",
    "Extra CLI flags (one per token, e.g. --highvram)",
    "multiline",
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Anima LoRA — native trainer")
        self.resize(1100, 800)
        self._options = backend.options()
        self._getters: dict[str, callable] = {}
        self._setters: dict[str, callable] = {}
        self._running = False

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_form_panel())
        splitter.addWidget(self._build_run_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._poll)
        self._timer.start()
        self._poll()

    # ----- form panel (scrollable curated fields) ------------------------- #
    def _build_form_panel(self) -> QWidget:
        inner = QWidget()
        vbox = QVBoxLayout(inner)
        for title, fields in _SECTIONS:
            vbox.addWidget(self._build_group(title, fields))
        # Extra-flags escape hatch (covers every non-curated train.py flag).
        gb = QGroupBox("Extra")
        form = QFormLayout(gb)
        dest, label, _ = _EXTRA_FIELD
        edit = QPlainTextEdit()
        edit.setMaximumHeight(70)
        edit.setPlaceholderText("--highvram\n--guidance_scale 1.0")
        self._getters[dest] = lambda e=edit: e.toPlainText().strip()
        self._setters[dest] = lambda v, e=edit: e.setPlainText(str(v or ""))
        form.addRow(label, edit)
        vbox.addWidget(gb)
        vbox.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        return scroll

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

    # ----- run panel (preview + controls + log) --------------------------- #
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
        return {dest: get() for dest, get in self._getters.items()}

    def _apply(self, form: dict) -> None:
        for dest, val in form.items():
            setter = self._setters.get(dest)
            if setter:
                setter(val)

    # ----- actions -------------------------------------------------------- #
    def _do_preview(self) -> None:
        try:
            cmd = backend.build_command(self._collect())
            self._preview.setPlainText(" ".join(cmd))
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
            elapsed = int(st.get("elapsed") or 0)
            self._status.setText(f"running · pid {st.get('pid')} · {elapsed}s")
        elif st.get("returncode") is not None:
            self._status.setText(f"finished · exit {st.get('returncode')}")
        else:
            self._status.setText("idle")
        lt = backend.log_tail(400)
        lines = lt.get("lines") or []
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
