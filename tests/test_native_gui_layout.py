from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from gui.native import app as native_app  # noqa: E402
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
