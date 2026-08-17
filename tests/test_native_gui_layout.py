from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from gui.native.app import MainWindow  # noqa: E402


def test_training_page_uses_kohya_style_accordion_flow() -> None:
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
            "Parameters / Advanced",
            "Samples",
            "Validation",
            "Metadata",
            "Monitoring",
            "Experimental",
            "Extra",
        ]
        assert set(window._opt_groups) == {"Sampling", "Validation"}
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
