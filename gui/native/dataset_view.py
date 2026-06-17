# -*- coding: utf-8 -*-
"""Dataset viewer/editor widget for the native panel (Utils → Dataset).

Browse an image folder (default ``image_dataset/``), see each image with its
``.txt`` caption side-by-side, edit + save the caption, sort the tags into the
Anima canonical keep-tokens order (``gui.native.tag_sort``) — for the open image OR
the **whole dataset** at once — and view **and paint** the SAM3/MIT mask
(``post_image_dataset/masks/…/{stem}_mask.png``).

Tag order (the kept head, before the ``|||`` separator) is::

    metadata → count (1girl…) → character → series → @artist → general

character / series classification is driven by the user-owned, swappable name lists
``dataset_tags/{characters,series}.txt`` (authoritative, override the vocab) plus the
anima-tagger ``vocab.json`` fallback.

The mask layer is held as a white-on-black ``QImage`` at image resolution — the
exact on-disk format — so loading an existing mask is a single ``drawPixmap`` and
saving is a single ``QImage.save`` (no per-pixel loops, no Qt6 alpha-channel
pitfalls). Brush paints white (masked) / black (erase); the overlay shows the
layer at partial opacity. (Red-tinted overlay is a cosmetic follow-up.)
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QImage, QImageReader, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import backend
from gui.native import image_dupes, tag_sort

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
        # Smooth (bilinear) scaling so the preview isn't pixelated/degraded when the
        # image is fit to the view — the default fast transform looks aliased.
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#0A0A0A"))
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
        # User-owned, swappable character / series name lists (authoritative).
        self._chars_path = Path(backend.ROOT) / tag_sort.DEFAULT_CHARACTERS_REL
        self._series_path = Path(backend.ROOT) / tag_sort.DEFAULT_SERIES_REL
        self._characters = tag_sort.load_name_set(self._chars_path)
        self._series = tag_sort.load_name_set(self._series_path)
        self._all_names: list[str] = []  # every image in the folder (filter source)
        self._cap_cache: dict[str, str] = {}  # name → lowercased caption (lazy)
        self._size_cache: dict[str, int] = {}  # name → pixel count (lazy, header-only)
        self._min_px = self._load_min_pixels()
        self._build()

    @staticmethod
    def _load_min_pixels() -> int:
        """The preprocess low-res threshold (configs/preprocess.toml) — reused for the
        'low-res' list filter. Falls back to 500k if the file/key is absent."""
        try:
            import tomllib

            p = Path(backend.ROOT) / "configs" / "preprocess.toml"
            if p.is_file():
                return int(tomllib.loads(p.read_text(encoding="utf-8"))["min_pixels"])
        except (OSError, ValueError, KeyError, ImportError):
            pass
        return 500_000

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        self._folder = QLineEdit(str(Path(backend.ROOT) / "image_dataset"))
        browse = QPushButton("📁")
        browse.setObjectName("icon")  # tight padding so the glyph isn't clipped
        browse.setFixedWidth(40)
        load = QPushButton("Load")
        browse.clicked.connect(self._choose_folder)
        load.clicked.connect(self._load_folder)
        top.addWidget(QLabel("Folder"))
        top.addWidget(self._folder, 1)
        top.addWidget(browse)
        top.addWidget(load)
        outer.addLayout(top)

        # filter / search bar
        flt = QHBoxLayout()
        self._filter_text = QLineEdit()
        self._filter_text.setPlaceholderText("filter… (tag or caption text)")
        self._filter_text.textChanged.connect(self._apply_filter)
        self._filter_mode = QComboBox()
        self._filter_mode.addItems(["caption contains", "tag present", "tag absent"])
        self._filter_mode.currentIndexChanged.connect(self._apply_filter)
        self._f_nomask = QCheckBox("no mask")
        self._f_nomask.toggled.connect(self._apply_filter)
        self._f_lowres = QCheckBox(f"low-res (<{self._min_px // 1000}k px)")
        self._f_lowres.toggled.connect(self._apply_filter)
        flt.addWidget(QLabel("🔍"))
        flt.addWidget(self._filter_text, 1)
        flt.addWidget(self._filter_mode)
        flt.addWidget(self._f_nomask)
        flt.addWidget(self._f_lowres)
        outer.addLayout(flt)

        # dataset hygiene: validate + near-duplicate grouping
        val = QHBoxLayout()
        b_validate = QPushButton("Validate dataset")
        b_validate.setToolTip(
            "Find captionless images, orphan .txt, and missing masks."
        )
        b_validate.clicked.connect(self._validate_dataset)
        b_dupes = QPushButton("Find duplicates")
        b_dupes.clicked.connect(self._find_duplicates)
        self._dup_thresh = QSpinBox()
        self._dup_thresh.setRange(0, 32)
        self._dup_thresh.setValue(10)
        self._dup_thresh.setToolTip("Max Hamming distance (bits) — lower = stricter")
        val.addWidget(b_validate)
        val.addWidget(b_dupes)
        val.addWidget(QLabel("max distance"))
        val.addWidget(self._dup_thresh)
        val.addStretch(1)
        outer.addLayout(val)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_list_panel())
        split.addWidget(self._build_center())
        split.addWidget(self._build_caption_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        outer.addWidget(split, 1)
        self._load_folder()

    def _build_list_panel(self) -> QWidget:
        panel = QWidget()
        lv = QVBoxLayout(panel)
        lv.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget()
        # Ctrl-click toggles / Shift-click ranges → multi-select (delete many at once).
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.currentItemChanged.connect(self._on_select)
        lv.addWidget(self._list, 1)
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color:#9aa4b2;")
        lv.addWidget(self._count_label)
        b_del = QPushButton("🗑 Delete selected (+caption+mask)")
        b_del.setToolTip(
            "Remove the image AND its .txt caption and mask together. "
            "Ctrl/Shift-click to select multiple."
        )
        b_del.clicked.connect(self._delete_selected)
        lv.addWidget(b_del)
        return panel

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
        rv.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Caption</b> (.txt)"))
        head.addStretch(1)
        b_save = QPushButton("💾 Save caption")
        b_save.clicked.connect(self._save_caption)
        head.addWidget(b_save)
        rv.addLayout(head)
        self._caption = QPlainTextEdit()
        self._caption.setPlaceholderText("Select an image to edit its caption…")
        rv.addWidget(self._caption, 1)
        rv.addWidget(self._build_tagtools_group())
        rv.addWidget(self._build_autocaption_group())
        rv.addWidget(self._build_bulktags_group())
        rv.addWidget(self._build_taggui_group())
        return right

    def _build_taggui_group(self) -> QGroupBox:
        box = QGroupBox("External: TagGUI")
        v = QVBoxLayout(box)
        v.setSpacing(6)
        hint = QLabel(
            "Launch jhc13/taggui in a separate window (its own full tag/caption tool). "
            "Point this at your taggui checkout; it reopens its last folder."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa4b2;")
        v.addWidget(hint)
        row = QHBoxLayout()
        self._taggui_dir = QLineEdit(os.environ.get("TAGGUI_DIR", ""))
        self._taggui_dir.setPlaceholderText("path to taggui checkout (or TAGGUI_DIR)")
        row.addWidget(self._taggui_dir, 1)
        b_open = QPushButton("Open TagGUI")
        b_open.clicked.connect(self._open_taggui)
        row.addWidget(b_open)
        v.addLayout(row)
        return box

    def _open_taggui(self) -> None:
        res = backend.run_taggui({"taggui_dir": self._taggui_dir.text().strip()})
        if not res.get("ok"):
            QMessageBox.warning(self, "TagGUI", str(res.get("error") or res))
            return
        QMessageBox.information(
            self,
            "TagGUI",
            "TagGUI launched in a separate window.\n"
            "In TagGUI, open this dataset folder (File → Load Directory).",
        )

    def _build_autocaption_group(self) -> QGroupBox:
        box = QGroupBox("Auto-caption (Qwen)")
        v = QVBoxLayout(box)
        v.setSpacing(6)
        hint = QLabel(
            "Captions the <b>selected</b> images (Ctrl/Shift-click) with the model in "
            "dataset_tags/qwen_caption.toml. Runs in the background — watch output/logs."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa4b2;")
        v.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(QLabel("mode"))
        self._cap_mode = QComboBox()
        self._cap_mode.addItems(["tags", "natural"])
        row.addWidget(self._cap_mode)
        self._cap_overwrite = QCheckBox("overwrite existing")
        row.addWidget(self._cap_overwrite)
        row.addStretch(1)
        b_cap = QPushButton("✦ Caption selected")
        b_cap.clicked.connect(self._caption_selected)
        row.addWidget(b_cap)
        v.addLayout(row)
        return box

    def _build_bulktags_group(self) -> QGroupBox:
        box = QGroupBox("Bulk tag edit + stats")
        v = QVBoxLayout(box)
        v.setSpacing(6)
        hint = QLabel(
            "Add / remove / replace tags across the <b>selected</b> images' captions. "
            "Comma-separated; replace uses <tt>old=new</tt> pairs. Case-insensitive."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa4b2;")
        v.addWidget(hint)
        self._bt_add = QLineEdit()
        self._bt_add.setPlaceholderText("add: tag1, tag2")
        self._bt_remove = QLineEdit()
        self._bt_remove.setPlaceholderText("remove: tag1, tag2")
        self._bt_replace = QLineEdit()
        self._bt_replace.setPlaceholderText("replace: old=new, old2=new2")
        for w in (self._bt_add, self._bt_remove, self._bt_replace):
            v.addWidget(w)
        row = QHBoxLayout()
        b_stats = QPushButton("📊 Tag stats (folder)")
        b_stats.clicked.connect(self._show_tag_stats)
        row.addWidget(b_stats)
        row.addStretch(1)
        b_apply = QPushButton("Apply to selected")
        b_apply.clicked.connect(self._bulk_edit_selected)
        row.addWidget(b_apply)
        v.addLayout(row)
        return box

    @staticmethod
    def _parse_replace(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for chunk in text.split(","):
            if "=" in chunk:
                k, val = chunk.split("=", 1)
                if k.strip() and val.strip():
                    out[k.strip()] = val.strip()
        return out

    def _bulk_edit_selected(self) -> None:
        from library.captioning import tag_stats as ts

        items = self._list.selectedItems()
        if not items or self._dir is None:
            QMessageBox.information(
                self, "Bulk tag edit", "Select one or more images (Ctrl/Shift-click)."
            )
            return
        add = ts.split_tags(self._bt_add.text())
        remove = ts.split_tags(self._bt_remove.text())
        replace = self._parse_replace(self._bt_replace.text())
        if not (add or remove or replace):
            QMessageBox.information(
                self, "Bulk tag edit", "Fill add / remove / replace first."
            )
            return
        names = [it.text() for it in items]
        paths = ts.caption_paths_for(self._dir, names)
        res = ts.bulk_edit_captions(paths, add=add, remove=remove, replace=replace)
        for n in names:  # filter cache reads lowercased text — invalidate touched ones
            self._cap_cache.pop(n, None)
        self._on_select(self._list.currentItem(), None)  # refresh shown caption
        QMessageBox.information(
            self,
            "Bulk tag edit",
            f"Changed {res['changed']} / {res['scanned']} caption(s); "
            f"{res['skipped']} skipped (no .txt).",
        )

    def _show_tag_stats(self) -> None:
        from library.captioning import tag_stats as ts

        if self._dir is None or not self._all_names:
            QMessageBox.information(self, "Tag stats", "Load a folder first.")
            return
        counts = ts.tag_frequencies(ts.caption_paths_for(self._dir, self._all_names))
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Tag stats — {self._dir.name}")
        dlg.resize(380, 540)
        lay = QVBoxLayout(dlg)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setStyleSheet("font-family: monospace;")
        view.setPlainText(ts.format_stats(counts))
        lay.addWidget(view)
        close = QPushButton("Close")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close)
        dlg.exec()

    def _caption_selected(self) -> None:
        items = self._list.selectedItems()
        if not items or self._dir is None:
            QMessageBox.information(
                self, "Auto-caption", "Select one or more images (Ctrl/Shift-click)."
            )
            return
        images = [str(self._dir / it.text()) for it in items]
        res = backend.run_qwen_caption(
            {
                "images": images,
                "mode": self._cap_mode.currentText(),
                "overwrite": self._cap_overwrite.isChecked(),
            }
        )
        if not res.get("ok"):
            QMessageBox.warning(self, "Auto-caption", str(res.get("error") or res))
            return
        QMessageBox.information(
            self,
            "Auto-caption",
            f"Captioning {len(images)} image(s) started in the background.\n"
            "When the log shows it finished, reselect an image to see its caption.",
        )

    def _build_tagtools_group(self) -> QGroupBox:
        box = QGroupBox("Tag order — keep tokens")
        v = QVBoxLayout(box)
        v.setSpacing(6)
        order = QLabel(
            "Head order: <b>metadata → count → character → series → @artist</b> → general"
        )
        order.setWordWrap(True)
        order.setStyleSheet("color:#9aa4b2;")
        v.addWidget(order)
        self._keep_sep = QCheckBox(
            f"Insert keep-tokens separator ({tag_sort.KEEP_TOKENS_SEPARATOR}) after @artist"
        )
        self._keep_sep.setChecked(True)
        v.addWidget(self._keep_sep)
        btns = QHBoxLayout()
        b_sort = QPushButton("Sort current")
        b_sort.setToolTip("Reorder the open image's caption.")
        b_sort_all = QPushButton("⮃ Sort ALL in dataset")
        b_sort_all.setToolTip("Reorder every .txt caption in the loaded folder.")
        b_sort_all.setObjectName("primary")
        b_sort.clicked.connect(self._sort_caption)
        b_sort_all.clicked.connect(self._sort_all_captions)
        btns.addWidget(b_sort)
        btns.addWidget(b_sort_all, 1)
        v.addLayout(btns)

        names = QGroupBox("Character / series lists")
        nv = QVBoxLayout(names)
        nv.setSpacing(4)
        self._cls_status = QLabel()
        self._cls_status.setWordWrap(True)
        nv.addWidget(self._cls_status)
        row = QHBoxLayout()
        b_chars = QPushButton("Characters file…")
        b_series = QPushButton("Series file…")
        b_reload = QPushButton("↻ Reload")
        b_chars.clicked.connect(lambda: self._pick_name_file("characters"))
        b_series.clicked.connect(lambda: self._pick_name_file("series"))
        b_reload.clicked.connect(self._reload_name_lists)
        row.addWidget(b_chars)
        row.addWidget(b_series)
        row.addWidget(b_reload)
        nv.addLayout(row)
        v.addWidget(names)
        self._update_cls_status()
        return box

    # ----- character / series name lists ---------------------------------- #
    def _update_cls_status(self) -> None:
        vocab_msg = "vocab.json" if self._vocab else "rule-based"
        self._cls_status.setText(
            f"classifier: <b>{vocab_msg}</b> · "
            f"characters: <b>{len(self._characters)}</b> "
            f"({self._chars_path.name}) · "
            f"series: <b>{len(self._series)}</b> ({self._series_path.name})"
        )

    def _pick_name_file(self, kind: str) -> None:
        start = self._chars_path if kind == "characters" else self._series_path
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {kind} list (one name per line)",
            str(start.parent if start.parent.is_dir() else backend.ROOT),
            "Text (*.txt);;All files (*)",
        )
        if not path:
            return
        if kind == "characters":
            self._chars_path = Path(path)
        else:
            self._series_path = Path(path)
        self._reload_name_lists()

    def _reload_name_lists(self) -> None:
        self._characters = tag_sort.load_name_set(self._chars_path)
        self._series = tag_sort.load_name_set(self._series_path)
        self._update_cls_status()

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
        d = Path(self._folder.text().strip())
        self._dir = d
        self._all_names = []
        self._cap_cache.clear()
        self._size_cache.clear()
        if d.is_dir():
            self._all_names = [
                f.name
                for f in sorted(d.iterdir())
                if f.is_file() and f.suffix.lower() in _IMG_EXTS
            ]
        self._apply_filter()

    # ----- filter / search ------------------------------------------------ #
    def _caption_text(self, name: str) -> str:
        """Lowercased caption text for ``name`` (cached; '' when no .txt)."""
        if name in self._cap_cache:
            return self._cap_cache[name]
        txt = (self._dir / name).with_suffix(".txt") if self._dir else None
        s = ""
        if txt is not None and txt.exists():
            try:
                s = txt.read_text(encoding="utf-8").lower()
            except OSError:
                s = ""
        self._cap_cache[name] = s
        return s

    def _pixels(self, name: str) -> int:
        """Pixel count via QImageReader (header only — no full decode); cached."""
        if name in self._size_cache:
            return self._size_cache[name]
        sz = QImageReader(str(self._dir / name)).size() if self._dir else QSize()
        px = sz.width() * sz.height() if sz.isValid() else 0
        self._size_cache[name] = px
        return px

    def _passes(self, name: str) -> bool:
        q = self._filter_text.text().strip().lower()
        if q:
            mode = self._filter_mode.currentText()
            cap = self._caption_text(name)
            if mode == "caption contains":
                if q not in cap:
                    return False
            else:  # tag present / tag absent — exact comma-token match
                present = any(t.strip() == q for t in cap.split(","))
                if mode == "tag present" and not present:
                    return False
                if mode == "tag absent" and present:
                    return False
        if self._f_nomask.isChecked() and self._mask_path(self._dir / name).exists():
            return False
        if self._f_lowres.isChecked() and self._pixels(name) >= self._min_px:
            return False
        return True

    def _apply_filter(self) -> None:
        if self._dir is None:
            return
        self._list.blockSignals(True)
        self._list.clear()
        shown = [n for n in self._all_names if self._passes(n)]
        self._list.addItems(shown)
        self._list.blockSignals(False)
        total = len(self._all_names)
        self._count_label.setText(
            f"{len(shown)} / {total} shown"
            if len(shown) != total
            else f"{total} images"
        )

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
            tag_sort.sort_caption(
                self._caption.toPlainText(),
                self._vocab,
                insert_sep=self._keep_sep.isChecked(),
                characters=self._characters,
                series=self._series,
            )
        )

    def _sort_all_captions(self) -> None:
        """Reorder every .txt caption in the loaded folder (dataset-wide scope)."""
        if self._dir is None or self._list.count() == 0:
            QMessageBox.information(self, "Sort all", "Load an image folder first.")
            return
        n = self._list.count()
        if (
            QMessageBox.question(
                self,
                "Sort ALL captions",
                f"Reorder tags in all {n} shown caption(s) in:\n{self._dir}\n"
                "(filter the list first to sort only a subset.)\n\n"
                "The .txt files are overwritten in place. Continue?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        insert_sep = self._keep_sep.isChecked()
        prog = QProgressDialog("Sorting captions…", "Cancel", 0, n, self)
        prog.setWindowModality(Qt.WindowModal)
        changed = missing = 0
        for i in range(n):
            if prog.wasCanceled():
                break
            prog.setValue(i)
            txt = (self._dir / self._list.item(i).text()).with_suffix(".txt")
            if not txt.exists():
                missing += 1
                continue
            try:
                orig = txt.read_text(encoding="utf-8")
            except OSError:
                continue
            new = tag_sort.sort_caption(
                orig,
                self._vocab,
                insert_sep=insert_sep,
                characters=self._characters,
                series=self._series,
            )
            if new.strip() != orig.strip():
                try:
                    txt.write_text(new.strip() + "\n", encoding="utf-8")
                    changed += 1
                except OSError:
                    pass
        prog.setValue(n)
        self._cap_cache.clear()  # captions changed on disk → drop stale filter cache
        # Refresh the open caption (it may have just been rewritten on disk).
        if self._current is not None:
            cur = self._current.with_suffix(".txt")
            if cur.exists():
                self._caption.setPlainText(cur.read_text(encoding="utf-8"))
        note = f"Sorted {changed} of {n} caption(s)."
        if missing:
            note += f" {missing} image(s) had no .txt."
        QMessageBox.information(self, "Sort all", note)

    def _save_caption(self) -> None:
        if self._current is None:
            return
        try:
            self._current.with_suffix(".txt").write_text(
                self._caption.toPlainText().strip() + "\n", encoding="utf-8"
            )
            self._cap_cache.pop(self._current.name, None)  # invalidate filter cache
        except OSError as exc:
            QMessageBox.warning(self, "Save caption", str(exc))

    # ----- safe delete (image + caption + mask together) ------------------ #
    def _delete_image(self, img: Path) -> int:
        """Delete the image and its .txt + mask. Returns the number of files removed."""
        removed = 0
        for p in (img, img.with_suffix(".txt"), self._mask_path(img)):
            try:
                if p.exists():
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        for cache in (self._cap_cache, self._size_cache):
            cache.pop(img.name, None)
        if img.name in self._all_names:
            self._all_names.remove(img.name)
        return removed

    def _delete_selected(self) -> None:
        items = self._list.selectedItems()
        if not items or self._dir is None:
            return
        imgs = [self._dir / it.text() for it in items]
        if len(imgs) == 1:
            extras = [
                p.name
                for p in (imgs[0].with_suffix(".txt"), self._mask_path(imgs[0]))
                if p.exists()
            ]
            tail = (" + " + " + ".join(extras)) if extras else ""
            msg = f"Delete {imgs[0].name}{tail}?"
        else:
            msg = f"Delete {len(imgs)} selected images (+ caption + mask each)?"
        if (
            QMessageBox.question(self, "Delete image", msg)
            != QMessageBox.StandardButton.Yes
        ):
            return
        for img in imgs:
            self._delete_image(img)
            if self._current == img:
                self._current = None
                self._view.set_image(None)
                self._caption.setPlainText("")
        self._apply_filter()

    # ----- dataset validation: near-duplicate detection ------------------- #
    @staticmethod
    def _dhash(path: Path, size: int = 8) -> int | None:
        """64-bit difference hash via QImage (no extra deps): downscale to grayscale
        (size+1 × size) and compare horizontally adjacent pixels."""
        img = QImage(str(path))
        if img.isNull():
            return None
        img = img.convertToFormat(QImage.Format_Grayscale8).scaled(
            size + 1, size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        bits = 0
        i = 0
        for y in range(size):
            for x in range(size):
                if img.pixelColor(x, y).red() > img.pixelColor(x + 1, y).red():
                    bits |= 1 << i
                i += 1
        return bits

    def _find_duplicates(self) -> None:
        if self._dir is None or self._list.count() == 0:
            return
        hashes: dict[str, int] = {}
        for i in range(self._list.count()):
            name = self._list.item(i).text()
            h = self._dhash(self._dir / name)
            if h is not None:
                hashes[name] = h
            if i % 25 == 0:
                QApplication.processEvents()  # keep the UI responsive on big folders
        groups = image_dupes.group_duplicates(hashes, self._dup_thresh.value())
        self._show_dupes(groups, hashes)

    def _show_dupes(self, groups: list[list[str]], hashes: dict[str, int]) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Duplicate groups")
        dlg.resize(460, 520)
        lv = QVBoxLayout(dlg)
        n = sum(len(g) for g in groups)
        lv.addWidget(
            QLabel(
                f"{len(groups)} group(s), {n} images. Double-click a file to open it."
                if groups
                else "No near-duplicates found at this threshold."
            )
        )
        tree = QTreeWidget()
        tree.setHeaderLabels(["file", "similarity"])
        for gi, g in enumerate(groups, 1):
            top = QTreeWidgetItem([f"Group {gi} ({len(g)} images)", ""])
            ref = hashes[g[0]]
            for name in g:
                sim = image_dupes.similarity(hashes[name], ref)
                child = QTreeWidgetItem([name, f"{sim * 100:.0f}%"])
                top.addChild(child)
            tree.addTopLevelItem(top)
            top.setExpanded(True)
        tree.itemDoubleClicked.connect(lambda it, _c: self._goto_image(it))
        lv.addWidget(tree, 1)
        if groups:
            b_resolve = QPushButton("Keep highest-res, delete the rest (+caption+mask)")
            b_resolve.clicked.connect(lambda: self._resolve_dupes(groups, dlg))
            lv.addWidget(b_resolve)
        dlg.exec()

    def _resolve_dupes(self, groups: list[list[str]], dlg: QDialog) -> None:
        """Keep the highest-resolution image in each group, safely delete the others."""
        losers = [
            n for g in groups for n in g if n != max(g, key=lambda x: self._pixels(x))
        ]
        if not losers:
            return
        if (
            QMessageBox.question(
                self,
                "Resolve duplicates",
                f"Delete {len(losers)} lower-res duplicate(s) across {len(groups)} "
                "group(s), keeping the highest-res of each (image + caption + mask)?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        for name in losers:
            if self._dir is not None:
                self._delete_image(self._dir / name)
        dlg.accept()
        self._apply_filter()
        QMessageBox.information(self, "Resolve duplicates", f"Deleted {len(losers)}.")

    def _goto_image(self, item: QTreeWidgetItem) -> None:
        if item.childCount():  # a group header, not a file
            return
        matches = self._list.findItems(item.text(0), Qt.MatchExactly)
        if matches:
            self._list.setCurrentItem(matches[0])

    # ----- dataset validation: orphans / captionless / missing masks ------ #
    def _validate_dataset(self) -> None:
        if self._dir is None or not self._dir.is_dir():
            QMessageBox.information(self, "Validate", "Load an image folder first.")
            return
        stems = {Path(n).stem for n in self._all_names}
        captionless, nomask = [], []
        for n in self._all_names:
            cap = (self._dir / n).with_suffix(".txt")
            if not cap.exists() or not self._caption_text(n).strip():
                captionless.append(n)
            if not self._mask_path(self._dir / n).exists():
                nomask.append(n)
        orphans = sorted(
            f.name
            for f in self._dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt" and f.stem not in stems
        )
        self._show_validation(captionless, orphans, nomask)

    def _show_validation(
        self, captionless: list[str], orphans: list[str], nomask: list[str]
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Dataset validation")
        dlg.resize(460, 540)
        lv = QVBoxLayout(dlg)
        total = len(self._all_names)
        clean = not (captionless or orphans or nomask)
        lv.addWidget(
            QLabel(
                f"{total} images — no issues found ✓"
                if clean
                else f"{total} images. Double-click an image row to open it."
            )
        )
        tree = QTreeWidget()
        tree.setHeaderLabels(["item", ""])
        for title, names in (
            (f"Captionless / empty .txt ({len(captionless)})", captionless),
            (f"Orphan .txt — no image ({len(orphans)})", orphans),
            (f"Missing mask ({len(nomask)})", nomask),
        ):
            top = QTreeWidgetItem([title, ""])
            for name in names:
                top.addChild(QTreeWidgetItem([name, ""]))
            top.setExpanded(bool(names) and len(names) <= 50)
            tree.addTopLevelItem(top)
        tree.itemDoubleClicked.connect(lambda it, _c: self._goto_image(it))
        lv.addWidget(tree, 1)
        if orphans:
            b_orphans = QPushButton(f"Delete {len(orphans)} orphan .txt")
            b_orphans.clicked.connect(lambda: self._delete_orphans(orphans, dlg))
            lv.addWidget(b_orphans)
        dlg.exec()

    def _delete_orphans(self, orphans: list[str], dlg: QDialog) -> None:
        if (
            QMessageBox.question(
                self, "Delete orphans", f"Delete {len(orphans)} caption file(s)?"
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        removed = 0
        for name in orphans:
            try:
                (self._dir / name).unlink()
                removed += 1
            except OSError:
                pass
        dlg.accept()
        QMessageBox.information(self, "Delete orphans", f"Deleted {removed}.")
