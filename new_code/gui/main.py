"""QT-based app using qasync and PyQt5
"""
import asyncio
import time
from functools import partial

from qasync import QEventLoop, asyncSlot, asyncClose
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import (
    pyqtSignal,
    QObject,
    QThread,
    QTimer,
    pyqtSlot,
    Qt
)
import numpy as np
import yaml

from core.app import AppController
from core.listen import CHUNK
from core.modes import (
    CONTINUOUS,
    INDEPENDENT,
    SYNCHRONIZED,
    TRIGGERED,
)
from gui.ui.mainview import Ui_MainWindow
from gui.ui.streamview import Ui_StreamView
from gui.ui.utils import QHLine
from gui.plotwidgets import SpectrogramWidget, WaveformWidget


SHOW_DURATION = 3.0


class QAppController(QObject, AppController):
    RECV = pyqtSignal([int, object])  # Receive data on stream index
    SETUP_STREAMS = pyqtSignal(object)
    DETECTED = pyqtSignal([int])  # Report detection on stream index

    def apply_config(self, config):
        super().apply_config(config)
        self.stream_controller.OUTPUT.connect(self.RECV.emit)
        self.stream_controller.DETECTED.connect(self.DETECTED.emit)
        self.SETUP_STREAMS.emit(config["streams"])


class StreamViewWidget(widgets.QWidget):
    def __init__(self, stream_info: dict, device_info: dict, app):
        super().__init__()

        self.app = app
        self.stream_info = stream_info
        self._device_info = device_info
        self.init_ui()

        self.connect_events()
        self.apply_stream_info()

    def init_ui(self):
        self.ui = Ui_StreamView()
        self.ui.setupUi(self)

        self.spec_view = SpectrogramWidget(
            CHUNK,
            min_freq=500,
            max_freq=10000,
            window=SHOW_DURATION,
            show_x=False, #True if ch_idx == self.channels - 1 else False,
            cmap=None,
        )
        self.amp_view = WaveformWidget(
            CHUNK,
            show_x=False,
            window=SHOW_DURATION,
        )
        self.amp_view.setMinimumHeight(20)

        if self.ui.spectrogramViewCheckBox.isChecked():
            self.spec_view.show()
        else:
            self.spec_view.hide()

        if self.ui.amplitudeViewCheckBox.isChecked():
            self.amp_view.show()
        else:
            self.amp_view.hide()

        self.ui.plotViewLayout.addWidget(self.spec_view, 3)
        self.ui.plotViewLayout.addWidget(self.amp_view, 1)

    def on_synchronized(self, synchronized: bool):
        print(synchronized)
        hide_on_synchronized = [
            self.ui.triggeredButton,
            self.ui.continuousButton,
            self.ui.recordButton,
            self.ui.monitorButton,
        ]
        for element in hide_on_synchronized:
            if synchronized:
                element.hide()
            else:
                element.show()

    def connect_events(self):
        self.ui.amplitudeViewCheckBox.clicked.connect(self.on_amplitude_checkbox)
        self.ui.spectrogramViewCheckBox.clicked.connect(self.on_spectrogram_checkbox)

        self.ui.gainSpinner.valueChanged.connect(self.on_gain_change)
        self.ui.thresholdSpinner.valueChanged.connect(self.on_threshold_change)

    def on_amplitude_checkbox(self):
        if self.ui.amplitudeViewCheckBox.isChecked():
            self.amp_view.show()
        else:
            self.amp_view.hide()

    def on_spectrogram_checkbox(self):
        if self.ui.spectrogramViewCheckBox.isChecked():
            self.spec_view.show()
        else:
            self.spec_view.hide()

    def on_gain_change(self, new_value):
        self.app.stream_controller.set_stream_gain(self.stream_info["idx"], new_value)

    def on_threshold_change(self, new_value):
        self.app.stream_controller.set_stream_threshold(self.stream_info["idx"], new_value)
        self.amp_view.set_threshold(new_value)

    def apply_stream_info(self):
        self.ui.indexLabel.setText(str(self.stream_info["idx"] + 1))
        self.ui.streamNameLabel.setText(self.stream_info["name"])

        self.ui.channelDropdown.clear()
        max_channels = self._device_info.get("max_input_channels")
        for i in range(max_channels):
            self.ui.channelDropdown.addItem(str(i + 1), i + 1)
        self.ui.channelDropdown.setCurrentIndex(self.stream_info["channel"])

        self.ui.triggeredButton.setChecked(self.stream_info["triggered"])
        self.ui.continuousButton.setChecked(not self.stream_info["triggered"])
        self.ui.monitorButton.setChecked(self.stream_info["monitor"])
        self.ui.recordButton.setChecked(not self.stream_info["monitor"])
        self.ui.gainSpinner.setValue(self.stream_info["gain"])
        self.ui.thresholdSpinner.setValue(self.stream_info["threshold"])

    def draw(self):
        self.spec_view.draw()
        self.amp_view.draw()

    @pyqtSlot(object)
    def receive_data(self, data):
        self.spec_view.receive_data(data)
        self.amp_view.receive_data(data)


class MainWindow(widgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Recorder"
        self.stream_views = []

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Container for main application logic
        self.app = QAppController()

        self.connect_events()

        self.on_refresh_device_list()

        self.app.RECV.connect(self.on_receive_data)
        # self.app.SETUP_STREAMS.connect(self.on_setup_streams)
        # self.app.DETECTED.connect(self.update_detections)
        #
        # self.label_texts = []
        # self.label_detects = []

        self.frame_timer = QTimer()
        self.frame_timer.start(int(1000 / 20))
        self.frame_timer.timeout.connect(self._loop)

    def _loop(self):
        for stream_view in self.stream_views:
            stream_view.draw()

    def on_receive_data(self, stream_idx: int, data):
        self.stream_views[stream_idx].receive_data(data[:, 0])

    @asyncClose
    async def closeEvent(self, event):
        self.app.stop()
        return

    def connect_events(self):
        self.ui.synchronizedModeButton.clicked.connect(partial(self.on_synchronized, True))
        self.ui.independentModeButton.clicked.connect(partial(self.on_synchronized, False))

        self.ui.recordButton.clicked.connect(self.on_record_all)
        self.ui.monitorButton.clicked.connect(self.on_monitor_all)

        self.ui.triggeredModeButton.clicked.connect(partial(self.on_triggered, TRIGGERED))
        self.ui.continuousModeButton.clicked.connect(partial(self.on_triggered, CONTINUOUS))

        self.ui.deviceDropdown.currentIndexChanged.connect(self.on_select_device)
        self.ui.openDeviceButton.clicked.connect(self.on_open_device)

    def on_refresh_device_list(self):
        device_list = self.app.list_devices()
        input_devices = [device for device in device_list if device["max_input_channels"] > 0]
        device_names = [device["name"] for device in input_devices]

        last_device = self.ui.deviceDropdown.currentData()
        self.ui.deviceDropdown.clear()

        for i, device in enumerate(input_devices):
            self.ui.deviceDropdown.addItem(device["name"], device)
            if last_device and device["name"] == last_device["name"]:
                self.ui.deviceDropdown.setCurrentIndex(i)

    def on_select_device(self, idx):
        device = self.ui.deviceDropdown.currentData()
        self.ui.channelDropdown.clear()
        max_channels = device.get("max_input_channels")
        for i in range(max_channels):
            self.ui.channelDropdown.addItem(str(i + 1), i + 1)

    def generate_config(self, device, channels):
        """Generate a config dict from the current selected options"""
        config = {}
        config["device_name"] = device["name"]
        config["synchronized"] = self.ui.synchronizedModeButton.isChecked()
        # config["gain"] = 0
        config["save"] = {
            "triggered": True,
            "min_file_duration": 2.0,
            "max_file_duration": 10.0,
            "buffer_duration": 0.5,
            "base_dir": "temp",
            "subdirectories": ["{name}", "{date}", "{hour}"],
            "filename_format": "{name}_{timestamp}",

        }
        config["detect"] = {"threshold": 1000.0}
        config["streams"] = []

        for ch in range(channels):
            config["streams"].append({
                "channel": ch,
                "name": "Ch{}".format(ch)
            })
        return config

    def on_open_device(self):
        device = self.ui.deviceDropdown.currentData()
        channels = self.ui.channelDropdown.currentData()

        # Here we could look up old configs for that device.
        config = self.generate_config(device, channels)
        self.set_config(config)

        self.set_synchronized_view_state(config.get("synchronized", False))

    def set_synchronized_view_state(self, synchronized):
        for stream_view in self.stream_views:
            stream_view.apply_stream_info()
            stream_view.on_synchronized(synchronized)

        if synchronized:
            self.ui.recordButton.setText("Record")
            self.ui.monitorButton.setText("Monitor")
        else:
            self.ui.recordButton.setText("Record\nAll")
            self.ui.monitorButton.setText("Monitor\nAll")

    def on_synchronized(self, synchronized: bool):
        self.app.set_synchronized(synchronized)

        for stream_view in self.stream_views:
            stream_view.apply_stream_info()
            stream_view.on_synchronized(synchronized)

        self.set_synchronized_view_state(synchronized)
        self.app.run()

    def on_record_all(self):
        self.app.stream_controller.set_monitor(False)
        self.refresh_stream_infos()

    def on_monitor_all(self):
        self.app.stream_controller.set_monitor(True)
        self.refresh_stream_infos()

    def refresh_stream_infos(self):
        for i, stream_info in enumerate(self.app.stream_controller.get_streams()):
            self.stream_views[i].stream_info = stream_info
            self.stream_views[i].apply_stream_info()

    def on_triggered(self, mode):
        self.app.stream_controller.set_recording_mode(mode)

    def set_config(self, config):
        self.app.apply_config(config)
        self.app.stream_controller.set_monitor(True)
        self.app.run()

        for i in reversed(range(self.ui.streamViewLayout.count())):
            self.ui.streamViewLayout.itemAt(i).widget().setParent(None)

        device = self.ui.deviceDropdown.currentData()
        self.stream_views = []
        for stream in self.app.stream_controller.get_streams():
            stream_view = StreamViewWidget(stream, device, self.app)
            self.ui.streamViewLayout.addWidget(stream_view, 1)
            self.ui.streamViewLayout.addWidget(QHLine())
            self.stream_views.append(stream_view)


def run_app(config_file):
    import sys

    with open(config_file, "r") as yaml_config:
        config = yaml.load(yaml_config)
    print(config["devices"])

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    mainWindow = MainWindow()
    # mainWindow.set_config(config["devices"][0])
    mainWindow.show()

    with loop:
        loop.run_forever()
