"""QT-based app using qasync and PyQt5
"""
import asyncio
import datetime
import os
import time
from functools import partial

from qasync import QEventLoop, asyncSlot, asyncClose
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import (
    pyqtSignal,
    QObject,
    QSettings,
    QThread,
    QTimer,
    pyqtSlot,
    Qt
)
import numpy as np
import yaml

from core.app import AppController
from core.config import RecordingConfig, PlotConfig
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
from gui.ui.preferenceview import Ui_PreferenceView
from gui.ui.utils import QHLine
from gui.plotwidgets import SpectrogramWidget, WaveformWidget


SHOW_DURATION = 3.0
DEFAULT_SAVE_FILE = "settings.json"


class QAppController(AppController, QObject):
    """A QObject version of the headless core.app.AppController

    The multiple inheritance requires the AppController to apply its
    __getattr__ override first before QObject's.
    """
    RECV = pyqtSignal([int, object])  # Receive data on stream index
    SETUP_STREAMS = pyqtSignal(object)
    DETECTED = pyqtSignal([int])  # Report detection on stream index
    MIC_SETUP = pyqtSignal(object)

    def apply_config(self, config):
        super().apply_config(config)
        self.stream_controller.OUTPUT.connect(self.RECV.emit)
        self.stream_controller.DETECTED.connect(self.DETECTED.emit)
        self.SETUP_STREAMS.emit(config["streams"])
        self.stream_controller.mic.SETUP.connect(self.MIC_SETUP.emit)


class PreferenceViewWidget(widgets.QDialog, Ui_PreferenceView):

    def __init__(self, config):
        super().__init__()
        self.setupUi(self)
        self.config = config
        self.config_cls = type(config)
        self.ordered_items = list(self.config.items())
        self.item_types = [type(v) for _, v in self.ordered_items]

        self.inputs = []
        for key, val in self.ordered_items:
            layout = widgets.QHBoxLayout()
            layout.addWidget(widgets.QLabel(key, self))
            self.inputs.append(widgets.QLineEdit(str(val), self))
            layout.addWidget(self.inputs[-1])
            self.optionsLayout.addLayout(layout)

    def accept(self):
        to_set = {}
        for line, (key, old_val), type_ in zip(self.inputs, self.ordered_items, self.item_types):
            try:
                value = type_(line.text())
            except:
               err_msg = widgets.QMessageBox()
               err_msg.setIcon(widgets.QMessageBox.Critical)
               err_msg.setText("Invalid parameter for {}: {}".format(key, line.text()))
               err_msg.setWindowTitle("Invalid parameter")
               err_msg.setStandardButtons(widgets.QMessageBox.Ok)
               err_msg.exec()
               return
            else:
                to_set[key] = value
        for key, val in to_set.items():
            self.config[key] = val

        super().accept()


class StreamViewWidget(widgets.QWidget):
    def __init__(self, stream_info: dict, device_info: dict, app, plot_config: dict):
        super().__init__()

        self.app = app
        self.stream_info = stream_info
        self._device_info = device_info
        self.plot_config = plot_config
        self._last_detect = None
        self.init_ui()

        self.connect_events()
        self.apply_stream_info()

    def init_ui(self):
        self.ui = Ui_StreamView()
        self.ui.setupUi(self)

        self.spec_view = SpectrogramWidget(self.plot_config)
        self.amp_view = WaveformWidget(self.plot_config)
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
        self.ui.channelDropdown.activated.connect(self.on_channel_select)
        self.ui.editNameButton.clicked.connect(self.on_edit_name)

    def on_edit_name(self):
        value, okay = widgets.QInputDialog.getText(
                self,
                "Subject/Channel Name",
                "Name",
                widgets.QLineEdit.Normal,
                self.ui.streamNameLabel.text())
        if okay and value:
            self.ui.streamNameLabel.setText(value)
            self.ui.streamNameLabel.setToolTip(value)
            self.app.set_stream_name(self.stream_info["idx"], value)

    def on_record(self):
        self.app.set_stream_monitor(self.stream_info["idx"], False)

    def on_monitor(self):
        self.app.set_stream_monitor(self.stream_info["idx"], True)

    def on_triggered(self, mode):
        self.app.set_stream_recording_mode(self.stream_info["idx"], mode)

    def on_amplitude_checkbox(self):
        if self.ui.amplitudeViewCheckBox.isChecked():
            self.amp_view.show()
        else:
            self.amp_view.hide()

    def on_setup_stream(self, setup):
        chunk = setup.get("chunk")
        rate = setup.get("rate")
        self.spec_view.set_info(chunk=chunk, rate=rate)
        self.amp_view.set_info(chunk=chunk, rate=rate)

    def on_detect(self):
        self._last_detect = time.time()

    def on_spectrogram_checkbox(self):
        if self.ui.spectrogramViewCheckBox.isChecked():
            self.spec_view.show()
        else:
            self.spec_view.hide()

    def on_gain_change(self, new_value):
        self.app.set_stream_gain(self.stream_info["idx"], new_value)

    def on_channel_select(self, channel):
        if not self.app.mic:
            return

        new_channel = self.ui.channelDropdown.currentData()
        if new_channel is not None:
            self.app.set_stream_channel(self.stream_info["idx"], new_channel)

    def on_threshold_change(self, new_value):
        self.app.set_stream_threshold(self.stream_info["idx"], new_value)
        self.amp_view.set_threshold(new_value)

        self.ui.thresholdSpinner.setValue(int(new_value))
        self.ui.thresholdSlider.setValue(int(new_value))

    def apply_stream_info(self):
        self.ui.indexLabel.setText(str(self.stream_info["idx"] + 1))
        self.ui.streamNameLabel.setText(self.stream_info["name"])

        self.ui.channelDropdown.clear()
        max_channels = self._device_info.get("max_input_channels")
        for i in range(max_channels):
            self.ui.channelDropdown.addItem(str(i + 1), i)
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
        if (
                self._last_detect is not None and
                time.time() - self._last_detect < self.app.get_buffer_duration()
                ):
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
        self._last_valid_path_inputs = {}

        self.init_ui()

        # Container for main application logic
        self.app = QAppController()
        self.settings = QSettings("Theunissen Lab", "MultipleMics")

        self.connect_events()

        self.on_refresh_device_list()

        self.app.RECV.connect(self.on_receive_data)
        self.app.DETECTED.connect(self.on_detect)
        self.app.MIC_SETUP.connect(self.on_setup_streams)


        self.plot_config = PlotConfig()
        # self.app.DETECTED.connect(self.update_detections)
        #
        # self.label_texts = []
        # self.label_detects = []

        self.frame_timer = QTimer()
        self.frame_timer.start(int(1000 / 20))
        self.frame_timer.timeout.connect(self._loop)

        if self.settings.value("lastConfig"):
            self.set_config(self.settings.value("lastConfig"))

    def init_ui(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.subdirectoryFormatInput.setText("{name},{date},{hour}")
        self.ui.filenameFormatInput.setText("{name}_{timestamp}")
        self.ui.baseSaveFolderInput.setText("temp")

        self.render_example_path()

    def _loop(self):
        for stream_view in self.stream_views:
            stream_view.draw()
            self.render_example_path()

    def on_detect(self, stream_idx: int):
        self.stream_views[stream_idx].on_detect()

    def on_receive_data(self, stream_idx: int, data):
        self.stream_views[stream_idx].receive_data(data[:, 0])

    def on_setup_streams(self, setup_dict):
        for stream_view in self.stream_views:
            stream_view.on_setup_stream(setup_dict)

    @asyncClose
    async def closeEvent(self, event):
        config = self.app.to_config()
        self.settings.setValue("lastConfig", config)
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

        self.ui.saveConfigAction.triggered.connect(self.on_save_config)
        self.ui.loadConfigAction.triggered.connect(self.on_load_config)

        self.ui.baseSaveFolderInput.clicked.connect(self.on_select_base_save_folder)

        self.ui.plotParametersAction.triggered.connect(self.on_plot_parameters)

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
        self.app.apply_collector_config(self.collect_inputs_to_dict())

    def generate_config(self, device: dict, channels: int):
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

        return RecordingConfig(config)

    def on_open_device(self):
        device = self.ui.deviceDropdown.currentData()
        channels = self.ui.channelDropdown.currentData()

        # Try to preserve stream names when possible when switching devices
        last_stream_infos = self.app.get_streams()
        # Here we could look up old configs for that device.
        config = self.generate_config(device, channels)
        for new_stream_config, old_stream_info in zip(config["streams"], last_stream_infos):
            new_stream_config["name"] = old_stream_info["name"]
        self.set_config(config)
        self.set_synchronized_view_state(config.get("synchronized", False))

    def set_synchronized_view_state(self, synchronized: bool):
        self.refresh_stream_infos()
        for stream_view in self.stream_views:
            stream_view.on_synchronized(synchronized)

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
        self.app.set_monitor(False)
        self.refresh_stream_infos()

    def on_monitor_all(self):
        self.app.set_monitor(True)
        self.refresh_stream_infos()

    def refresh_stream_infos(self):
        for i, stream_info in enumerate(self.app.get_streams()):
            self.stream_views[i].stream_info = stream_info
            self.stream_views[i].apply_stream_info()

    def on_triggered(self, mode: str):
        self.app.set_recording_mode(mode)
        self.refresh_stream_infos()

    def set_config(self, config: RecordingConfig):
        self.on_refresh_device_list()
        for i in range(self.ui.deviceDropdown.count()):
            if self.ui.deviceDropdown.itemData(i)["name"] == config["device_name"]:
                self.ui.deviceDropdown.setCurrentIndex(i)
                break

        self.app.apply_config(config)
        self.app.set_monitor(True)
        self.app.run()

        for i in reversed(range(self.ui.streamViewLayout.count())):
            self.ui.streamViewLayout.itemAt(i).widget().setParent(None)

        device = self.ui.deviceDropdown.currentData()
        self.stream_views = []
        for stream in self.app.get_streams():
            stream_view = StreamViewWidget(stream, device, self.app, self.plot_config)
            self.ui.streamViewLayout.addWidget(stream_view, 1)
            self.ui.streamViewLayout.addWidget(QHLine())
            self.stream_views.append(stream_view)

        self.set_synchronized_view_state(config["synchronized"])
        self.ui.triggeredModeButton.setChecked(config["collect.triggered"])
        self.ui.continuousModeButton.setChecked(not config["collect.triggered"])

        self.ui.subdirectoryFormatInput.setText(",".join(config["save.subdirectories"]))
        self.ui.filenameFormatInput.setText(config["save.filename_format"])
        self.ui.baseSaveFolderInput.setText(config["save.base_dir"])

        self.render_example_path()

    def path_inputs_to_dict(self):
        save_dict = {
            "subdirectories": self.ui.subdirectoryFormatInput.text().split(","),
            "filename_format": self.ui.filenameFormatInput.text(),
            "base_dir": self.ui.baseSaveFolderInput.text(),
        }
        try:
            _saver = Pathfinder()
            _saver.apply_config(save_dict)
        except ValueError:
            return self._last_valid_path_inputs
        else:
            self._last_valid_path_inputs = save_dict
            return save_dict

    def render_example_path(self):
        _saver = Pathfinder()
        _saver.apply_config(self.path_inputs_to_dict())
        self.ui.exampleSavePathLabel.setText(
            "Example: {}".format(_saver.get_save_path("STREAM", datetime.datetime.now()))
        )

    def on_save_config(self):
        options = widgets.QFileDialog.Options()
        save_file, _ = widgets.QFileDialog.getSaveFileName(
            self,
            "Save current settings to config file",
            DEFAULT_SAVE_FILE,
            "*",
            options=options)

        if save_file:
            self.app.save_config(save_file)

    def on_load_config(self):
        options = widgets.QFileDialog.Options()
        file_name, _ = widgets.QFileDialog.getOpenFileName(
            self,
            "Load config file",
            DEFAULT_SAVE_FILE,
            "*",
            options=options)

        if file_name:
            self.set_config(RecordingConfig.read_json(file_name))

    def on_select_base_save_folder(self):
        file_name = widgets.QFileDialog.getExistingDirectory(
            self,
            "Select base save folder",
            os.path.expanduser("~"),
            widgets.QFileDialog.ShowDirsOnly,
        )

        if file_name:
            self.ui.baseSaveFolderInput.setText(file_name)

    def on_plot_parameters(self):
        dialog = PreferenceViewWidget(self.plot_config)
        result = dialog.exec_()
        if result:
            for stream_view in self.stream_views:
                stream_view.spec_view.apply_config(self.plot_config)
                stream_view.amp_view.apply_config(self.plot_config)


def run_app(config_file=None):
    import sys

    app = widgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    mainWindow = MainWindow()
    mainWindow.show()

    if config_file:
        with open(config_file, "r") as yaml_config:
            config = yaml.load(yaml_config, Loader=yaml.FullLoader)
            mainWindow.set_config(RecordingConfig(config["devices"][0]))

    with loop:
        loop.run_forever()
