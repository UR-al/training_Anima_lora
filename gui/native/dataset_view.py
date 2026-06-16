# -*- coding: utf-8 -*-
"""Dataset viewer/editor widget for the native panel (Utils → Dataset).

Browse an image folder (default ``image_dataset/``), see each image with its
``.txt`` caption side-by-side, edit + save the caption, sort the tags into the
Anima canonical order (``gui.native.tag_sort``), and overlay the SAM3/MIT mask
(``post_image_dataset/masks/…/{stem}_mask.png``) with a toggle.

Milestone: viewer + caption edit/sort + mask overlay (read-only). Brush painting
of the mask is the next milestone — :class:`ImageMaskView` already keeps the mask
as a separate layer so the paint path slots in without a rewrite.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import backend
from gui.native import tag_sort

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_MASKS_REL = "post_image_dataset/masks"


class ImageMaskView(QWidget):
    """Aspect-fit image with an optional tinted mask overlay."""

    def __init__(self) -> None:
        super().__init__()
        self._pix: QPixmap | None = None
        self._mask: QPixmap | None = None
        self._show_mask = False
        self.setMinimumSize(360, 360)

    def set_image(self, pix: QPixmap | None) -> None:
        self._pix = pix
        self.update()

    def set_mask(self, pix: QPixmap | None) -> None:
        self._mask = pix
        self.update()

    def set_show_mask(self, on: bool) -> None:
        self._show_mask = on
        self.update()

    def has_mask(self) -> bool:
        return self._mask is not None

    def _fit_rect(self, size: QSize) -> QRect:
        if size.isEmpty():
            return self.rect()
        scale = min(self.width() / size.width(), self.height() / size.height())
        w, h = int(size.width() * scale), int(size.height() * scale)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.darkGray)
        if self._pix is None:
            return
        target = self._fit_rect(self._pix.size())
        p.drawPixmap(target, self._pix)
        if self._show_mask and self._mask is not None:
            p.setOpacity(0.45)
            p.drawPixmap(target, self._mask)
            p.setOpacity(1.0)


class DatasetView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._dir: Path | None = None
        self._current: Path | None = None
        self._vocab = tag_sort.load_vocab_categories(
            Path(backend.ROOT) / tag_sort.DEFAULT_VOCAB_REL
        )
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        self._folder = QLineEdit(str(Path(backend.ROOT) / "image_dataset"))
        browse = QPushButton("📁")
        browse.setFixedWidth(36)
        load = QPushButton("Load")
        browse.clicked.connect(self._choose_folder)
        load.clicked.connect(self._load_folder)
        top.addWidget(QLabel("Folder"))
        top.addWidget(self._folder, 1)
        top.addWidget(browse)
        top.addWidget(load)
        outer.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        split.addWidget(self._list)

        center = QWidget()
        cv = QVBoxLayout(center)
        self._view = ImageMaskView()
        cv.addWidget(self._view, 1)
        self._mask_toggle = QCheckBox("Show mask overlay")
        self._mask_toggle.toggled.connect(self._view.set_show_mask)
        cv.addWidget(self._mask_toggle)
        self._mask_note = QLabel("")
        cv.addWidget(self._mask_note)
        split.addWidget(center)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("Caption (.txt)"))
        self._caption = QPlainTextEdit()
        rv.addWidget(self._caption, 1)
        btns = QHBoxLayout()
        b_sort = QPushButton("Sort tags")
        b_save = QPushButton("Save caption")
        b_sort.clicked.connect(self._sort_caption)
        b_save.clicked.connect(self._save_caption)
        btns.addWidget(b_sort)
        btns.addWidget(b_save)
        rv.addLayout(btns)
        vocab_msg = "vocab.json loaded" if self._vocab else "no vocab.json (rule-based)"
        rv.addWidget(QLabel(f"tag classifier: {vocab_msg}"))
        split.addWidget(right)

        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        outer.addWidget(split, 1)
        self._load_folder()

    # ----- folder / list -------------------------------------------------- #
    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select image folder",
            self._folder.text().strip() or str(backend.ROOT),
        )
        if path:
            self._folder.setText(path)
            self._load_folder()

    def _load_folder(self) -> None:
        self._list.clear()
        d = Path(self._folder.text().strip())
        self._dir = d
        if not d.is_dir():
            return
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                self._list.addItem(f.name)

    def _mask_path(self, image: Path) -> Path:
        rel = image.name
        if self._dir is not None:
            try:
                rel = str(image.relative_to(self._dir))
            except ValueError:
                rel = image.name
        rel_path = Path(rel)
        return (
            Path(backend.ROOT) / _MASKS_REL / rel_path.parent / f"{image.stem}_mask.png"
        )

    def _on_select(self, current, _prev) -> None:
        if current is None or self._dir is None:
            return
        img = self._dir / current.text()
        self._current = img
        self._view.set_image(QPixmap(str(img)) if img.exists() else None)
        mask = self._mask_path(img)
        if mask.exists():
            self._view.set_mask(QPixmap(str(mask)))
            self._mask_note.setText(f"mask: {mask.name}")
        else:
            self._view.set_mask(None)
            self._mask_note.setText("no mask for this image")
        cap = img.with_suffix(".txt")
        self._caption.setPlainText(
            cap.read_text(encoding="utf-8") if cap.exists() else ""
        )

    # ----- caption -------------------------------------------------------- #
    def _sort_caption(self) -> None:
        self._caption.setPlainText(
            tag_sort.sort_caption(self._caption.toPlainText(), self._vocab)
        )

    def _save_caption(self) -> None:
        if self._current is None:
            return
        try:
            self._current.with_suffix(".txt").write_text(
                self._caption.toPlainText().strip() + "\n", encoding="utf-8"
            )
        except OSError as exc:
            QMessageBox.warning(self, "Save caption", str(exc))
