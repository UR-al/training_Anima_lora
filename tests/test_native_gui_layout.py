from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from gui.native import app as native_app  # noqa: E402
from gui import backend  # noqa: E402
from gui.native.app import MainWindow  # noqa: E402
from gui.modules.config_io import load_toml_to_form, save_form_to_toml  # noqa: E402


def test_training_page_uses_kohya_style_accordion_flow(monkeypatch) -> None:
    monkeypatch.setattr(native_app, "_LANG", "ko")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        assert isinstance(window._training_scroll, QScrollArea)
        assert window._training_inner is None
        assert list(window._sections) == [
            "Configuration",
            "Source model",
            "Folders",
            "Dataset preparation",
            "Parameters",
            "Parameters / Basic",
            "Parameters / Network",
            "Parameters / Optimizer",
            "Parameters / Scheduler",
            "Parameters / Loss",
            "Parameters / Sigma low-resolution",
            "Parameters / Advanced",
            "Samples",
            "Validation",
            "Metadata",
            "Monitoring",
            "Experimental",
            "Extra",
        ]
        assert set(window._opt_groups) == {"Sampling", "Validation"}
        assert (
            window._field_sections["sigma_lowres"]
            is window._sections["Parameters / Sigma low-resolution"]
        )
        assert not window._widgets["sigma_lowres_route"].isEnabled()
        assert "먼저 켜야" in window._widgets["sigma_lowres_route"].toolTip()
        window._widgets["sigma_lowres"].setChecked(True)
        window._widgets["sigma_lowres_route"].setText("1024:896")
        app.processEvents()
        assert window._widgets["sigma_lowres_route"].isEnabled()
        assert "해상도 전환 경로" in window._widgets["sigma_lowres_route"].toolTip()

        saved = save_form_to_toml(window._collect(for_save=True))
        loaded = load_toml_to_form(saved, known_dests=set(window._widgets))
        assert loaded["sigma_lowres"] is True
        assert loaded["sigma_lowres_route"] == "1024:896"

        window._widgets["sigma_lowres"].setChecked(False)
        window._widgets["sigma_lowres_route"].clear()
        window._apply(loaded)
        assert window._widgets["sigma_lowres"].isChecked()
        assert window._widgets["sigma_lowres_route"].text() == "1024:896"
    finally:
        window._timer.stop()
        window.close()
        app.processEvents()


def test_every_schema_argument_is_rendered_and_searchable() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        schema_dests = {
            arg.get("dest")
            for args in window._tab_schema.values()
            for arg in args
            if arg.get("dest")
        }
        assert window._rendered_schema == schema_dests

        searchable = {entry["dest"] for entry in window._build_search_index()}
        assert schema_dests <= searchable
        assert all(window._field_sections.get(dest) for dest in schema_dests)
    finally:
        window._timer.stop()
        window.close()
        app.processEvents()


def test_lycoris_direct_preset_and_commented_alternatives() -> None:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    try:
        custom = window._widgets["lycoris_preset_custom"]
        assert not custom.isEnabled()

        text = """
network_module = "lycoris.kohya"
network_train_unet_only = true
network_args = [
  "algo=lokr",
  "preset=C:/presets/custom anima.toml",
  "factor=4",
]
optimizer_type = "LoraEasyCustomOptimizer.ademamix.SimplifiedAdEMAMix"
# optimizer_type = "LoraEasyCustomOptimizer.came.CAME"
optimizer_args = [
  "betas=(0.99, 0.99)",
  "weight_decay=0.0",
# "betas=(0.9,0.999,0.9999)",
# "foreach=True",
]
"""
        form = load_toml_to_form(text, known_dests=set(window._widgets))
        comments = form.pop("_commented_settings")
        window._apply(form)
        window._set_ignored_settings(comments, window._collect(for_save=True))
        app.processEvents()

        assert window._getters["network_module"]() == "networks.lycoris_anima"
        assert window._scope.currentIndex() == 1
        assert window._getters["algo"]() == "lokr"
        assert window._getters["lycoris_preset"]() == backend.LYCORIS_CUSTOM_PRESET
        assert custom.isEnabled()
        assert window._getters["lycoris_preset_custom"]() == (
            "C:/presets/custom anima.toml"
        )
        command = backend.build_command(window._collect())
        assert "preset=C:/presets/custom anima.toml" in command
        assert "algo=lokr" in command
        assert "--network_train_unet_only" in command

        assert len(window._ignored_rows) == 3
        optimizer_row, betas_row, foreach_row = window._ignored_rows
        assert not any(row["check"].isChecked() for row in window._ignored_rows)

        betas_row["check"].setChecked(True)
        foreach_row["check"].setChecked(True)
        app.processEvents()
        args = window._getters["optimizer_args"]().splitlines()
        assert "betas=(0.9,0.999,0.9999)" in args
        assert "weight_decay=0.0" in args
        assert "foreach=True" in args

        betas_row["check"].setChecked(False)
        app.processEvents()
        args = window._getters["optimizer_args"]().splitlines()
        assert "betas=(0.99,0.99)" in args
        assert "weight_decay=0.0" in args and "foreach=True" in args

        optimizer_row["check"].setChecked(True)
        app.processEvents()
        assert (
            window._getters["optimizer_type"]() == "LoraEasyCustomOptimizer.came.CAME"
        )
        optimizer_row["check"].setChecked(False)
        app.processEvents()
        assert window._getters["optimizer_type"]().endswith("SimplifiedAdEMAMix")
    finally:
        window._timer.stop()
        window.close()
        app.processEvents()
