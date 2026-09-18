#!/usr/bin/env python3
"""Q3 - Interactive microswitch detector and classifier.

Run ``python q3_app.py``.  Open a Part1 scene image, click *Run Detection*,
then click a labelled box (or an item in the switch list) to classify it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import PyQt *before* OpenCV.  The pip OpenCV wheels ship a separate Qt
# plugin directory and set QT_QPA_PLATFORM_PLUGIN_PATH on import.  If that
# directory wins, PyQt's QApplication tries to load OpenCV's incompatible
# xcb plugin and aborts at startup.  Pin the plugin path back to PyQt below.
from PyQt5.QtCore import QLibraryInfo, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

_PYQT_PLUGIN_PATH = QLibraryInfo.location(QLibraryInfo.PluginsPath)

import cv2
import numpy as np

# cv2 modifies these variables at import time.  They must refer to PyQt's Qt
# runtime because this process creates the QApplication with PyQt5.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _PYQT_PLUGIN_PATH
os.environ.pop("QT_QPA_FONTDIR", None)

from q1_detect_contours import BOX_COLORS, detect_switches
from q2_classify_type import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, classify_image


class ImageView(QGraphicsView):
    """Graphics view that reports image-space coordinates and accepts drops."""

    image_clicked = pyqtSignal(QPointF)
    file_dropped = pyqtSignal(str)

    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None) -> None:
        super().__init__(scene, parent)
        self._zoom_steps = 0
        self._minimum_zoom_steps = -8
        self._maximum_zoom_steps = 20
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setAcceptDrops(True)

    def mousePressEvent(self, event: Any) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.image_clicked.emit(self.mapToScene(event.pos()))
        super().mousePressEvent(event)

    def wheelEvent(self, event: Any) -> None:
        """Zoom around the pointer; use Fit Image to reset this view."""

        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        if delta > 0 and self._zoom_steps < self._maximum_zoom_steps:
            self.scale(1.15, 1.15)
            self._zoom_steps += 1
        elif delta < 0 and self._zoom_steps > self._minimum_zoom_steps:
            self.scale(1 / 1.15, 1 / 1.15)
            self._zoom_steps -= 1
        event.accept()

    def dragEnterEvent(self, event: Any) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()

    def show_entire_image(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        rect = scene.itemsBoundingRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_steps = 0


class SwitchInspector(QMainWindow):
    """Desktop workflow joining Q1 bounding regions with the Q2 classifier."""

    def __init__(self, model_path: Path, train_dir: Path) -> None:
        super().__init__()
        self.model_path = model_path
        self.train_dir = train_dir
        self.image: Optional[np.ndarray] = None
        self.image_path: Optional[Path] = None
        self.detections: List[Dict[str, Any]] = []
        self.box_items: Dict[int, QGraphicsRectItem] = {}
        self.selected_id: Optional[int] = None

        self.setWindowTitle("Microswitch Inspector — Q3")
        self.resize(1280, 760)
        self._build_ui()
        self._show_status("Open a Part1 image to begin.")

    def _build_ui(self) -> None:
        self.scene = QGraphicsScene(self)
        self.view = ImageView(self.scene, self)
        self.view.setBackgroundBrush(QColor("#30343b"))
        self.view.image_clicked.connect(self.select_at)
        self.view.file_dropped.connect(self.load_image)

        self.open_button = QPushButton("Open Image…")
        self.open_button.clicked.connect(self.open_image)
        self.detect_button = QPushButton("Run Detection")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(self.run_detection)
        self.fit_button = QPushButton("Fit Image")
        self.fit_button.clicked.connect(self.view.show_entire_image)

        self.file_label = QLabel("No image loaded")
        self.file_label.setWordWrap(True)
        self.switch_list = QListWidget()
        self.switch_list.currentItemChanged.connect(self.select_from_list)

        self.id_value = QLabel("—")
        self.type_value = QLabel("—")
        self.confidence_value = QLabel("—")
        self.message_value = QLabel("Run detection, then click a labelled region.")
        self.message_value.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(self.open_button)
        controls.addWidget(self.detect_button)
        controls.addWidget(self.fit_button)
        controls.addStretch(1)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addLayout(controls)
        side_layout.addWidget(QLabel("Image"))
        side_layout.addWidget(self.file_label)
        side_layout.addWidget(QLabel("Detected switches"))
        side_layout.addWidget(self.switch_list, 1)
        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.StyledPanel)
        form = QFormLayout(result_frame)
        form.addRow("Selected ID:", self.id_value)
        form.addRow("Type:", self.type_value)
        form.addRow("Confidence:", self.confidence_value)
        form.addRow("Status:", self.message_value)
        side_layout.addWidget(QLabel("Classification result"))
        side_layout.addWidget(result_frame)

        splitter = QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([900, 380])
        self.setCentralWidget(splitter)

    def open_image(self) -> None:
        start_dir = str(self.image_path.parent if self.image_path else Path.cwd())
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open scene image", start_dir, "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if filename:
            self.load_image(filename)

    def load_image(self, filename: str) -> None:
        path = Path(filename)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self._error(f"Could not read image:\n{path}")
            return
        self.image, self.image_path = image, path
        self.detections = []
        self.selected_id = None
        self.file_label.setText(str(path))
        self.detect_button.setEnabled(True)
        self._clear_result("Image loaded. Click Run Detection.")
        self._redraw_scene()
        self._show_status(f"Loaded {path.name}")

    def run_detection(self) -> None:
        if self.image is None:
            return
        try:
            self.detections, _mask, _roi = detect_switches(self.image)
        except Exception as exc:
            self._error(f"Detection failed:\n{exc}")
            return
        self.selected_id = None
        self._clear_result(
            f"Found {len(self.detections)} switch(es). Click a box or list item."
        )
        self._redraw_scene()
        self._show_status(f"Detected {len(self.detections)} switch(es).")

    def _redraw_scene(self) -> None:
        self.scene.clear()
        self.box_items.clear()
        self.switch_list.blockSignals(True)
        self.switch_list.clear()
        self.switch_list.blockSignals(False)
        if self.image is None:
            return

        height, width = self.image.shape[:2]
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # ``tobytes`` is accepted by both QImage and PyQt's type stubs.
        # The QImage owns a copy of this data, so it remains valid after this
        # method returns.
        qimage = QImage(
            rgb.tobytes(),
            width,
            height,
            width * 3,
            QImage.Format.Format_RGB888,
        )
        self.scene.addItem(QGraphicsPixmapItem(QPixmap.fromImage(qimage)))

        for detection in self.detections:
            switch_id = int(detection["id"])
            x, y, w, h = detection["bbox"]
            b, g, r = BOX_COLORS[(switch_id - 1) % len(BOX_COLORS)]
            color = QColor(r, g, b)
            rect = QGraphicsRectItem(x, y, w, h)
            rect.setPen(QPen(color, 4))
            rect.setData(0, switch_id)
            self.scene.addItem(rect)
            self.box_items[switch_id] = rect
            label = QGraphicsSimpleTextItem(f"SW{switch_id}")
            label.setBrush(color)
            label.setPos(x, max(0, y - 22))
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            self.scene.addItem(label)
            item = QListWidgetItem(f"SW{switch_id}  |  x={x}, y={y}, w={w}, h={h}")
            item.setData(Qt.ItemDataRole.UserRole, switch_id)
            self.switch_list.addItem(item)

        self.scene.setSceneRect(0, 0, width, height)
        self.view.show_entire_image()

    def select_at(self, point: QPointF) -> None:
        if not self.detections:
            self._clear_result("No detections yet. Click Run Detection first.")
            return
        matches = []
        for detection in self.detections:
            x, y, w, h = detection["bbox"]
            if x <= point.x() <= x + w and y <= point.y() <= y + h:
                matches.append(detection)
        if not matches:
            self.selected_id = None
            self._set_box_highlight(None)
            self._clear_result("No switch selected: click inside a labelled bounding box.")
            self.switch_list.clearSelection()
            self._show_status("No switch selected.")
            return
        # If regions overlap, favour the more specific (smallest) region.
        self.classify_detection(min(matches, key=lambda d: d["bbox"][2] * d["bbox"][3]))

    def select_from_list(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        switch_id = int(current.data(Qt.ItemDataRole.UserRole))
        detection = next((d for d in self.detections if d["id"] == switch_id), None)
        if detection is not None:
            self.classify_detection(detection, update_list=False)

    def classify_detection(self, detection: Dict[str, Any], update_list: bool = True) -> None:
        if self.image is None:
            return
        switch_id = int(detection["id"])
        x, y, w, h = (int(value) for value in detection["bbox"])
        crop = self.image[y : y + h, x : x + w].copy()
        self.selected_id = switch_id
        self._set_box_highlight(switch_id)
        if update_list:
            for row in range(self.switch_list.count()):
                list_item = self.switch_list.item(row)
                if list_item is None:
                    continue
                if int(list_item.data(Qt.ItemDataRole.UserRole)) == switch_id:
                    self.switch_list.setCurrentRow(row)
                    break
        try:
            self.message_value.setText("Classifying selected crop…")
            QApplication.processEvents()
            label, confidence, _probabilities = classify_image(
                crop, self.model_path, self.train_dir
            )
        except Exception as exc:
            self.type_value.setText("Error")
            self.confidence_value.setText("—")
            self.message_value.setText(f"Classification failed: {exc}")
            self._show_status("Classification failed.")
            return
        self.id_value.setText(f"SW{switch_id}")
        self.type_value.setText(label)
        self.confidence_value.setText(f"{confidence:.1%}")
        self.message_value.setText("Classification complete.")
        self._show_status(f"SW{switch_id}: {label} ({confidence:.1%})")

    def _set_box_highlight(self, selected_id: Optional[int]) -> None:
        for switch_id, item in self.box_items.items():
            b, g, r = BOX_COLORS[(switch_id - 1) % len(BOX_COLORS)]
            color = QColor(r, g, b)
            item.setPen(QPen(color if switch_id != selected_id else QColor("#ffff00"), 6 if switch_id == selected_id else 4))

    def _clear_result(self, message: str) -> None:
        self.id_value.setText("—")
        self.type_value.setText("—")
        self.confidence_value.setText("—")
        self.message_value.setText(message)

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "Microswitch Inspector", message)
        self._show_status(message.replace("\n", " "))

    def _show_status(self, message: str) -> None:
        """Update the optional QMainWindow status bar safely."""

        status_bar = self.statusBar()
        if status_bar is not None:
            status_bar.showMessage(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Q1 + Q2 microswitch inspector")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Q2 joblib model")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_DATA_DIR, help="Q2 training images")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("Microswitch Inspector")
    window = SwitchInspector(args.model, args.train_dir)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
