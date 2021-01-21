"""QT-based app using qasync and PyQt5
"""
import asyncio
import time

from qasync import QEventLoop, asyncSlot, asyncClose
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import (
    pyqtSignal,
    QObject,
    QThread,
    pyqtSlot,
    Qt
)
import numpy as np
import yaml

from core.app import AppController


class QAppController(QObject, AppController):
    RECV = pyqtSignal([int, object])  # Receive data on stream index
    SETUP_STREAMS = pyqtSignal(object)
    DETECTED = pyqtSignal([int])  # Report detection on stream index

    def apply_config(self, config):
        super().apply_config(config)
        self.stream_controller.OUTPUT.connect(self.RECV.emit)
        self.stream_controller.DETECTED.connect(self.DETECTED.emit)
        self.SETUP_STREAMS.emit(config["streams"])


class MainWindow(widgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Recorder"
        self._mic_task = None

        # Container for main application logic
        self.app = QAppController()
        self.init_ui()

        self.app.RECV.connect(self.update_label)
        self.app.SETUP_STREAMS.connect(self.on_setup_streams)
        self.app.DETECTED.connect(self.update_detections)

        self.label_texts = []
        self.label_detects = []

    @asyncClose
    async def closeEvent(self, event):
        self.app.stop()
        self._mic_task.cancel()
        return

    def init_ui(self):
        main_frame = widgets.QFrame(self)
        self.layout = widgets.QVBoxLayout()
        self.scroll_area = widgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        main_frame.setLayout(self.layout)
        self.detect_label = widgets.QLabel("Detecte", self)
        self.layout.addWidget(self.detect_label)

        self.label = widgets.QLabel("Poop", self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.scroll_area)

        self.btn = widgets.QPushButton("Push me ", self)
        self.btn.clicked.connect(self.app.stop)

        self.layout.addWidget(self.btn)
        self.setCentralWidget(main_frame)

    def update_label(self, stream_idx, data):
        self.label_texts[stream_idx] = "{} {} {}".format(stream_idx, np.min(data), np.max(data))
        self.update()

    def update(self):
        now = time.time()
        detects = [then and now - then < 0.2 for then in self.label_detects]
        parts = []
        for detect, txt in zip(detects, self.label_texts):
            parts.append("{} {}".format("+" if detect else "-", txt))

        self.label.setText("\n".join(parts))

    def update_detections(self, stream_idx):
        self.label_detects[stream_idx] = time.time()
        self.update()

    def on_setup_streams(self, stream_configs):
        self.label_texts = [""] * len(stream_configs)
        self.label_detects = [None] * len(stream_configs)

    def set_config(self, config):
        self.app.apply_config(config)
        self._mic_task = asyncio.create_task(self.app.mic.run())


def run_app(config_file):
    import sys

    with open(config_file, "r") as yaml_config:
        config = yaml.load(yaml_config)
    print(config["devices"])

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    mainWindow = MainWindow()
    mainWindow.set_config(config["devices"][0])
    mainWindow.show()

    with loop:
        loop.run_forever()
