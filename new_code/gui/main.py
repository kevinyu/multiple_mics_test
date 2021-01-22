"""QT-based app using qasync and PyQt5
"""
import asyncio
import datetime
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
from core.streams import Pathfinder
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
        self._last_detect = None
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
        self.ui.thresholdSlider.valueChanged.connect(self.on_threshold_change)

        self.ui.recordButton.clicked.connect(self.on_record)
        self.ui.monitorButton.clicked.connect(self.on_monitor)
        self.ui.triggeredButton.clicked.connect(partial(self.on_triggered, TRIGGERED))
        self.ui.continuousButton.clicked.connect(partial(self.on_triggered, CONTINUOUS))

    def on_record(self):
        self.app.stream_controller.set_stream_monitor(self.stream_info["idx"], False)

    def on_monitor(self):
        self.app.stream_controller.set_stream_monitor(self.stream_info["idx"], True)

    def on_triggered(self, mode):
        self.app.stream_controller.set_stream_recording_mode(self.stream_info["idx"], mode)

    def on_amplitude_checkbox(self):
        if self.ui.amplitudeViewCheckBox.isChecked():
            self.amp_view.show()
        else:
            self.amp_view.hide()

    def on_detect(self):
        self._last_detect = time.time()

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

        self.ui.thresholdSpinner.setValue(int(new_value))
        self.ui.thresholdSlider.setValue(int(new_value))

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
        if self._last_detect is not None and time.time() - self._last_detect < self.app.stream_controller.get_buffer_duration():
            self.ui.detectionIndicatorLabel.setText("Triggered")
        else:
            self.ui.detectionIndicatorLabel.setText("-")

    @pyqtSlot(object)
    def receive_data(self, data):
        self.spec_view.receive_data(data)
        self.amp_view.receive_data(data)


class MainWindow(widgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Recorder"
        self.stream_views = []

        self.init_ui()

        # Container for main application logic
        self.app = QAppController()

        self.connect_events()

        self.on_refresh_device_list()

        self.app.RECV.connect(self.on_receive_data)
        self.app.DETECTED.connect(self.on_detect)

        # self.app.SETUP_STREAMS.connect(self.on_setup_streams)
        # self.app.DETECTED.connect(self.update_detections)
        #
        # self.label_texts = []
        # self.label_detects = []

        self.frame_timer = QTimer()
        self.frame_timer.start(int(1000 / 20))
        self.frame_timer.timeout.connect(self._loop)

    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.subdirectoryFormatInput.setText("{name},{date},{hour}")
        self.ui.filenameFormatInput.setText("{name}_{timestamp}")
        self.ui.baseSaveFolderInput.setText("temp")

    def _loop(self):
        for stream_view in self.stream_views:
            stream_view.draw()

    def on_detect(self, stream_idx: int):
        self.stream_views[stream_idx].on_detect()

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
        self.ui.refreshDeviceListButton.clicked.connect(self.on_refresh_device_list)

        self.ui.deviceDropdown.currentIndexChanged.connect(self.on_select_device)
        self.ui.openDeviceButton.clicked.connect(self.on_open_device)

        self.ui.applyDetectConfigButton.clicked.connect(self.on_apply_collect_values)

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
        if device is None:
            return
        self.ui.channelDropdown.clear()
        max_channels = device.get("max_input_channels")
        for i in range(max_channels):
            self.ui.channelDropdown.addItem(str(i + 1), i + 1)

    def collect_inputs_to_dict(self):
        return {
            "triggered": self.ui.triggeredModeButton.isChecked(),
            "min_file_duration": self.ui.minDurationSpinner.value(),
            "max_file_duration": self.ui.maxDurationSpinner.value(),
            "buffer_duration": self.ui.bufferSpinner.value(),
        }

    def on_apply_collect_values(self):
        self.app.stream_controller.apply_collector_config(self.collect_inputs_to_dict())

    def generate_config(self, device, channels):
        """Generate a config dict from the current selected options"""
        config = {}
        config["device_name"] = device["name"]
        config["synchronized"] = self.ui.synchronizedModeButton.isChecked()
        config["gain"] = self.ui.defaultGainSpinner.value()
        config["collect"] = self.collect_inputs_to_dict()
        config["save"] = self.path_inputs_to_dict()
        config["detect"] = {"threshold": self.ui.defaultThresholdSpinner.value()}
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
        active_stream_trigger_modes = set()  # Check if the subviews have different trigger states

        self.refresh_stream_infos()

        for stream_view in self.stream_views:
            stream_view.on_synchronized(synchronized)
            active_stream_trigger_modes.add(stream_view.stream_info["triggered"])

        self.ui.synchronizedModeButton.setChecked(synchronized)
        self.ui.independentModeButton.setChecked(not synchronized)

        if synchronized:
            self.ui.recordButton.setText("Record")
            self.ui.monitorButton.setText("Monitor")
        else:
            self.ui.recordButton.setText("Record\nAll")
            self.ui.monitorButton.setText("Monitor\nAll")

    def on_synchronized(self, synchronized: bool):
        self.app.set_synchronized(synchronized)
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
        self.refresh_stream_infos()

    def set_config(self, config):
        # TODO make sure the device dropdown is set properly
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

        self.set_synchronized_view_state(config.get("synchronized", False))
        self.ui.triggeredModeButton.setChecked(config.get("collect", {}).get("triggered"))
        self.ui.continuousModeButton.setChecked(not config.get("collect", {}).get("triggered"))

        self.ui.subdirectoryFormatInput.setText(",".join(config.get("save", {}).get("subdirectories", [])))
        self.ui.filenameFormatInput.setText(config.get("save", {}).get("filename_format", "{name}_{timestamp}"))
        self.ui.baseSaveFolderInput.setText(config.get("save", {}).get("base_dir", "{name}_{timestamp}"))

        self.render_example_path()

    def path_inputs_to_dict(self):
        return {
            "subdirectories": self.ui.subdirectoryFormatInput.text().split(","),
            "filename_format": self.ui.filenameFormatInput.text(),
            "base_dir": self.ui.baseSaveFolderInput.text(),
        }

    def render_example_path(self):
        _saver = Pathfinder()
        _saver.apply_config(self.path_inputs_to_dict())
        self.ui.exampleSavePathLabel.setText(
            "Example: {}".format(_saver.get_save_path("STREAM", datetime.datetime.now()))
        )

def run_app(config_file=None):
    import sys

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    mainWindow = MainWindow()
    mainWindow.show()

    if config_file:
        with open(config_file, "r") as yaml_config:
            config = yaml.load(yaml_config)
            mainWindow.set_config(config["devices"][0])

    with loop:
        loop.run_forever()
