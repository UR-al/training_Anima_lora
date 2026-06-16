# -*- coding: utf-8 -*-
"""Dataset viewer/editor widget for the native panel (Utils → Dataset).

Browse an image folder (default ``image_dataset/``), see each image with its
``.txt`` caption side-by-side, edit + save the caption, sort the tags into the
Anima canonical order (``gui.native.tag_sort``), and view **and paint** the
SAM3/MIT mask (``post_image_dataset/masks/…/{stem}_mask.png``).

The mask layer is held as a white-on-black ``QImage`` at image resolution — the
exact on-disk format — so loading an existing mask is a single ``drawPixmap`` and
saving is a single ``QImage.save`` (no per-pixel loops, no Qt6 alpha-channel
pitfalls). Brush paints white (masked) / black (erase); the overlay shows the
layer at partial opacity. (Red-tinted overlay is a cosmetic follow-up.)
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
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
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from gui import backend
from gui.native import tag_sort

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_MASKS_REL = "post_image_dataset/masks"


class ImageMaskView(QWidget):
    """Aspect-fit image with a paintable white-on-black mask layer overlay."""

    def __init__(self) -> None:
        super().__init__()
        self._pix: QPixmap | None = None
        self._mask: QImage | None = None  # RGB32, image-res, white=masked
        self._show_mask = False
        self._editable = False
        self._dragging = False
        self.brush_size = 40  # image-pixel radius
        self.erase = False

    # ----- content -------------------------------------------------------- #
    def set_image(self, pix: QPixmap | None) -> None:
        self._pix = pix
        self._mask = None
        self.update()

    def load_mask(self, pix: QPixmap | None) -> None:
        if self._pix is None:
            self._mask = None
            self.update()
            return
        size = self._pix.size()
        m = QImage(size, QImage.Format_RGB32)
        m.fill(Qt.black)
        if pix is not None:
            p = QPainter(m)
            p.drawPixmap(QRect(0, 0, size.width(), size.height()), pix)
            p.end()
        self._mask = m
        self.update()

    def _ensure_mask(self) -> None:
        if self._mask is None and self._pix is not None:
            self._mask = QImage(self._pix.size(), QImage.Format_RGB32)
            self._mask.fill(Qt.black)

    def has_mask(self) -> bool:
        return self._mask is not None

    def clear_mask(self) -> None:
        self._ensure_mask()
        if self._mask is not None:
            self._mask.fill(Qt.black)
            self.update()

    def save_mask(self, path: Path) -> bool:
        if self._mask is None:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        return self._mask.save(str(path), "PNG")

    # ----- view / edit modes ---------------------------------------------- #
    def set_show_mask(self, on: bool) -> None:
        self._show_mask = on
        self.update()

    def set_editable(self, on: bool) -> None:
        self._editable = on
        if on:
            self._ensure_mask()
            self._show_mask = True
        self.update()

    # ----- geometry ------------------------------------------------------- #
    def _fit_rect(self, size: QSize) -> QRect:
        if size.isEmpty() or self.width() == 0 or self.height() == 0:
            return self.rect()
        scale = min(self.width() / size.width(), self.height() / size.height())
        w, h = int(size.width() * scale), int(size.height() * scale)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _img_point(self, pos) -> QPoint | None:
        if self._pix is None:
            return None
        tr = self._fit_rect(self._pix.size())
        if tr.width() == 0 or tr.height() == 0:
            return None
        x = (pos.x() - tr.x()) * self._pix.width() / tr.width()
        y = (pos.y() - tr.y()) * self._pix.height() / tr.height()
        return QPoint(int(x), int(y))

    def _paint_at(self, pos) -> None:
        self._ensure_mask()
        if self._mask is None:
            return
        ip = self._img_point(pos)
        if ip is None:
            return
        p = QPainter(self._mask)
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.black if self.erase else Qt.white)
        p.drawEllipse(ip, self.brush_size, self.brush_size)
        p.end()
        self.update()

    # ----- events --------------------------------------------------------- #
    def mousePressEvent(self, e) -> None:  # noqa: N802 (Qt override)
        if self._editable and e.button() == Qt.LeftButton:
            self._dragging = True
            self._paint_at(e.position().toPoint())

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._editable and self._dragging:
            self._paint_at(e.position().toPoint())

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        self._dragging = False

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.darkGray)
        if self._pix is None:
            return
        target = self._fit_rect(self._pix.size())
        p.drawPixmap(target, self._pix)
        if self._show_mask and self._mask is not None:
            p.setOpacity(0.45)
            p.drawImage(target, self._mask)
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
        split.addWidget(self._build_center())
        split.addWidget(self._build_caption_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        outer.addWidget(split, 1)
        self._load_folder()

    def _build_center(self) -> QWidget:
        center = QWidget()
        cv = QVBoxLayout(center)
        self._view = ImageMaskView()
        cv.addWidget(self._view, 1)
        # mask toolbar
        bar = QHBoxLayout()
        self._mask_toggle = QCheckBox("Show mask")
        self._mask_toggle.toggled.connect(self._view.set_show_mask)
        self._edit_toggle = QCheckBox("Edit (brush)")
        self._edit_toggle.toggled.connect(self._on_edit_toggle)
        self._erase = QCheckBox("Erase")
        self._erase.toggled.connect(lambda v: setattr(self._view, "erase", v))
        self._brush = QSpinBox()
        self._brush.setRange(1, 512)
        self._brush.setValue(self._view.brush_size)
        self._brush.valueChanged.connect(lambda v: setattr(self._view, "brush_size", v))
        bar.addWidget(self._mask_toggle)
        bar.addWidget(self._edit_toggle)
        bar.addWidget(self._erase)
        bar.addWidget(QLabel("brush"))
        bar.addWidget(self._brush)
        cv.addLayout(bar)
        bar2 = QHBoxLayout()
        b_clear = QPushButton("Clear mask")
        b_save = QPushButton("Save mask")
        b_clear.clicked.connect(self._view.clear_mask)
        b_save.clicked.connect(self._save_mask)
        bar2.addWidget(b_clear)
        bar2.addWidget(b_save)
        bar2.addStretch(1)
        self._mask_note = QLabel("")
        bar2.addWidget(self._mask_note)
        cv.addLayout(bar2)
        return center

    def _build_caption_panel(self) -> QWidget:
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
        return right

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
            self._view.load_mask(QPixmap(str(mask)))
            self._mask_note.setText(f"mask: {mask.name}")
        else:
            self._view.load_mask(None)
            self._mask_note.setText("no mask on disk")
        cap = img.with_suffix(".txt")
        self._caption.setPlainText(
            cap.read_text(encoding="utf-8") if cap.exists() else ""
        )

    def _on_edit_toggle(self, on: bool) -> None:
        self._view.set_editable(on)
        if on:
            self._mask_toggle.setChecked(True)

    def _save_mask(self) -> None:
        if self._current is None:
            return
        if not self._view.save_mask(self._mask_path(self._current)):
            QMessageBox.warning(
                self, "Save mask", "No mask to save (paint or load one)."
            )
        else:
            self._mask_note.setText(f"saved: {self._mask_path(self._current).name}")

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
