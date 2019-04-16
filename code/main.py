from __future__ import print_function
import os
import sys
from collections import defaultdict
from functools import partial

import numpy as np
import pyaudio
import sounddevice as sd
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread, pyqtSlot, Qt

from listeners import SoundDetector, SoundSaver
from plotwidgets import SpectrogramWidget, WaveformWidget
from settings import Settings


class BaseMicrophone(QObject):

    REC = pyqtSignal(object)
    STOP = pyqtSignal()

    def __init__(self, device_index=None, channels=1, parent=None):
        super(BaseMicrophone, self).__init__(parent)
        self.channels = channels
        self.device_index = device_index

    def _run(self):
        raise NotImplementedError

    def run(self):
        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.start()
        self._run()

    def set_channels(self, channels=1):
        if self._stream:
            self._stream.close()
        self.channels = channels
        self._run()

    def set_device_index(self, device_index=0):
        if self._stream:
            self._stream.close()
        self.device_index = device_index
        self._run()


class PyAudioMicrophone(BaseMicrophone):

    def __init__(self, device_index=None, channels=1, parent=None):
        super(PyAudioMicrophone, self).__init__(device_index=device_index, channels=channels, parent=parent)
        self._stream = None
        self.p = pyaudio.PyAudio()
        self._stop = False
        self._gain = Settings.GAIN

    def _run(self):
        def _callback(in_data, frame_count, time_info, status):
            if self._stop:
                return (None, pyaudio.paComplete)

            data = np.frombuffer(in_data, dtype=np.int16)[:]
            new_data = np.array([
                data[idx::self.channels]
                for idx in range(self.channels)
            ]).T
            scaled = np.power(10.0, Settings.GAIN[:self.channels] / 20.0) * new_data

            self.REC.emit(scaled)

            return (in_data, pyaudio.paContinue)

        try:
            self._stream = self.p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=Settings.RATE,
                frames_per_buffer=Settings.CHUNK,
                input=True,
                output=False,
                input_device_index=self.device_index,
                stream_callback=_callback,
            )
        except:
            return


class SoundDeviceMicrophone(BaseMicrophone):

    def __init__(self, device_index=None, channels=1, parent=None):
        super(SoundDeviceMicrophone, self).__init__(device_index=device_index, channels=channels, parent=parent)
        self._stream = None
        self._stop = False
        self._gain = Settings.GAIN

    def _run(self):
        def _callback(in_data, frame_count, time_info, status):
            scaled = np.power(10.0, Settings.GAIN[:self.channels] / 20.0) * in_data
            self.REC.emit(scaled.astype(np.int16))
            return

        try:
            self._stream = sd.InputStream(
                    dtype=np.int16,
                    samplerate=Settings.RATE,
                    blocksize=Settings.CHUNK,
                    device=self.device_index,
                    channels=self.channels,
                    callback=_callback)
        except:
            pass
        else:
            self._stream.start()


class MainWindow(widgets.QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.title = "Recorder"
        self.channels = 1

        self.program_controller = ProgramController(self)

        self.recording_window = RecordingWindow(self.channels, self)
        self.frame_timer = QTimer()
        self.frame_timer.start(1000 * 1.0/120.0)
        self.frame_timer.timeout.connect(self.recording_window._loop)
        self._old_threads = []

        self.mic = None
        self.channel_names = {}

        self.init_ui()
        self.setup_listeners()
        self.update_filename_format()
        self.update_display_path()
        self.connect_events()

        self._get_selected_mic(None)
        self.set_channels(self.channels)
        self.connect_mic()

    def closeEvent(self, *args, **kwargs):
        return

    def init_ui(self):
        main_frame = widgets.QFrame(self)
        self.layout = widgets.QVBoxLayout()

        self.recording_indicator = widgets.QLabel("Trigger OFF", self)

        self.scroll_area = widgets.QScrollArea(self)
        # self.scroll_area.setFixedHeight(650)
        # self.scroll_area.setFixedWidth(800)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.recording_window)

        self.layout.addWidget(self.program_controller)
        self.layout.addWidget(self.scroll_area)
        self.layout.addWidget(self.recording_indicator)
        main_frame.setLayout(self.layout)

        self.setCentralWidget(main_frame)

    def setup_listeners(self):
        """Instaniate listeners
        """

        self.saver = SoundSaver(
            size=Settings.RATE * Settings.FILE_DURATION,
            min_size=Settings.RATE * Settings.MIN_FILE_DURATION,
            filename_format=self._filename_format,
            path=Settings.get("SAVE_DIRECTORY"),
            triggered=False
        )
        self.saver.start()

        self.saver.SAVE_EVENT.connect(self.update_last_save)
        self.saver.RECORDING.connect(self.update_recording_indicator)

        self.detector = SoundDetector(
            size=int(Settings.RATE * Settings.DETECTION_WINDOW)
        )
        self.detector.start()

        self.detector.OUT.connect(self.saver.trigger)

    def update_last_save(self, path):
        self.program_controller.last_save.setText("Last saved {}".format(os.path.basename(path)))

    def update_recording_indicator(self, val):
        if val:
            self.recording_indicator.setText("Trigger ON")
        else:
            self.recording_indicator.setText("Trigger OFF")

    def _get_selected_mic(self, idx):
        device = self.program_controller.input_source.currentData()
        self.on_select_mic(device.get("index"))

    def on_select_mic(self, device_index):
        self.channels = 1
        self.on_recording_mode("monitor")
        self.program_controller.monitor_button.setChecked(True)
        if not self.mic:
            self.mic = SoundDeviceMicrophone(device_index=device_index, channels=self.channels)
        else:
            self.mic.set_device_index(device_index)

    def connect_mic(self):
        self.mic.REC.connect(self.recording_window.receive_data)
        self.mic.REC.connect(self.saver.receive_data)
        self.mic.REC.connect(self.detector.IN.emit)

    def on_select_channels(self, idx):
        self.set_channels(idx + 1)

    def set_channels(self, channels):
        if self.mic is None:
            return

        if len(Settings.GAIN) < channels:
            Settings.GAIN = np.concatenate([
                Settings.GAIN,
                np.zeros(channels - len(Settings.GAIN))
            ])
        self.mic.set_channels(channels)
        self.on_recording_mode("monitor")
        self.program_controller.monitor_button.setChecked(True)

        self.saver.reset()
        self.detector.reset()
        self.detector._channels = channels
        self.saver._channels = channels

        self.recording_window.set_channels(channels)

        for ch_idx in range(channels):
            (self.recording_window
                .controllers[ch_idx]
                .SET_THRESHOLD
                .connect(partial(self.detector.set_threshold, ch_idx)))
            (self.recording_window
                .controllers[ch_idx]
                .SET_GAIN
                .connect(partial(self.set_gain, ch_idx)))

    def set_gain(self, ch_idx, value):
        if len(Settings.GAIN) <= ch_idx:
            Settings.GAIN = np.concatenate([
                Settings.GAIN,
                np.zeros(1 + ch_idx - len(Settings.GAIN))
            ])

        Settings.GAIN[ch_idx] = value

    def on_recording_mode(self, mode):
        if mode == "monitor":
            self.saver.set_triggered(False)
            self.saver.set_saving(False)
            self.program_controller.last_save.setText("Monitoring...")
        elif mode == "triggered":
            self.saver.set_triggered(True)
            self.saver.set_saving(True)
            self.program_controller.last_save.setText("Recording on trigger...")
        elif mode == "continuous":
            self.saver.set_triggered(False)
            self.saver.set_saving(True)
            self.program_controller.last_save.setText("Recording...")

    def connect_events(self):
        self.program_controller.monitor_button.clicked.connect(
            partial(self.on_recording_mode, "monitor")
        )
        self.program_controller.trigger_button.clicked.connect(
            partial(self.on_recording_mode, "triggered")
        )
        self.program_controller.continuous_button.clicked.connect(
            partial(self.on_recording_mode, "continuous")
        )
        self.program_controller.save_button.clicked.connect(self.run_file_loader)

        self.program_controller.input_source.currentIndexChanged.connect(self._get_selected_mic)
        self.program_controller.input_source_channels.currentIndexChanged.connect(self.on_select_channels)

        self.recording_window.UPDATE_CHANNEL_NAMES.connect(self.on_channel_names_changed)
        self.recording_window.emit_channel_names()

    def update_display_path(self):
        path = self.saver.path
        if path is not None:
            display_path = str(os.path.join(
                os.sep,
                "[...]",
                os.path.basename(os.path.dirname(path)),
                os.path.basename(path),
                self.saver.filename_format
            ))
            self.program_controller.save_location.setToolTip("Saving to {}".format(display_path))
        else:
            display_path = os.path.join("[No path]", self.saver.filename_format)
            self.program_controller.save_location.setToolTip("Not saving to {} until path is set".format(display_path))

        text = "Saving to {}".format(display_path)
        if len(text) > 40:
            text = text[:37] + "..."
        self.program_controller.save_location.setText(text)

    def on_channel_names_changed(self, channel_names):
        Settings.set("CHANNEL_NAMES", channel_names)
        self.channel_names = channel_names

        self.update_filename_format()
        self.update_display_path()

    @property
    def _filename_format(self):
        channel_strings = []
        for key, val in sorted(self.channel_names.items()):
            channel_strings.append(
                "{}".format(val) if val else "ch{}".format(key)
            )

        return "{}_{{0}}.wav".format("_".join(channel_strings))

    def update_filename_format(self):
        self.saver.filename_format = self._filename_format

    def run_file_loader(self):
        options = widgets.QFileDialog.Options()
        path = widgets.QFileDialog.getExistingDirectory(
            self,
            "Save recordings to",
            self.saver.path or Settings.BASE_DIRECTORY,
            options=options)
        if path:
            Settings.set("SAVE_DIRECTORY", path)
            self.saver.path = path
            self.update_display_path()


class ProgramController(widgets.QFrame):

    def __init__(self, parent=None):
        super(ProgramController, self).__init__(parent=parent)
        self.p = pyaudio.PyAudio()
        self.init_ui()

    def init_ui(self):
        self.input_source = widgets.QComboBox(self)
        self.input_source_channels = widgets.QComboBox(self)
        self.save_button = widgets.QPushButton("Set save location", self)
        self.save_button.setToolTip("Save new recordings to this folder")
        self.save_location = widgets.QLabel("[No path]", self)
        self.last_save = widgets.QLabel("", self)

        self.monitor_button = widgets.QRadioButton("Monitor only")
        self.monitor_button.setToolTip("Do not save new files")
        self.monitor_button.setChecked(True)
        self.trigger_button = widgets.QRadioButton(
                "Triggered recording")
        self.trigger_button.setToolTip("Record files when sound amplitude triggers detector")
        self.continuous_button = widgets.QRadioButton("Continuous recording")
        self.continuous_button.setToolTip("Record audio continuously")

        if Settings.USE_SOUNDDEVICE:
            for i, device in enumerate(sd.query_devices()):
                if not device["max_input_channels"]:
                    continue
                device["index"] = i
                self.input_source.addItem(device.get("name"), device)
        else:
            for i in range(self.p.get_device_count()):
                device = self.p.get_device_info_by_index(i)
                if not device.get("maxInputChannels"):
                    continue
                self.input_source.addItem(device.get("name"), device)

        self.input_source.currentIndexChanged.connect(self.update_channel_dropdown)
        self.update_channel_dropdown(None)

        layout = widgets.QGridLayout()

        layout.setColumnStretch(1, 0.5)
        layout.setColumnStretch(2, 0.5)
        layout.addWidget(widgets.QLabel("Device:", parent=self), 1, 1)
        layout.addWidget(self.input_source, 1, 2)
        layout.addWidget(widgets.QLabel("Channels:", parent=self), 2, 1)
        layout.addWidget(self.input_source_channels, 2, 2)

        # layout.setColumnStretch(3, 1)
        layout.addWidget(self.monitor_button, 3, 2)
        layout.addWidget(self.trigger_button, 4, 2)
        layout.addWidget(self.continuous_button, 5, 2)

        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)
        layout.addWidget(self.save_button, 1, 3)
        layout.addWidget(self.save_location, 2, 3, 1, 2)
        layout.addWidget(self.last_save, 3, 3, 1, 2)

        self.setLayout(layout)

    def update_channel_dropdown(self, idx):
        device = self.input_source.currentData()
        self.input_source_channels.clear()
        
        if Settings.USE_SOUNDDEVICE:
            max_channels = device.get("max_input_channels")
        else:
            max_channels = device.get("maxInputChannels")

        for i in range(max_channels):
            self.input_source_channels.addItem(str(i + 1), i + 1)


class RecordingController(widgets.QFrame):

    SET_THRESHOLD = pyqtSignal(int)
    SET_GAIN = pyqtSignal(int)
    SET_NAME = pyqtSignal(str)

    def __init__(self, idx=None, name=None, parent=None):
        """
        Control settings for recording on this channel

        i.e. gain and threshold
        """
        super(RecordingController, self).__init__(parent=parent)

        self.idx = idx
        self.name = name

        self.name_button = widgets.QPushButton(self.short_name, self)
        self.name_button.setToolTip(self.display_name)
        # self.gain_control = widgets.QDoubleSpinBox(self)
        self.gain_title = widgets.QLabel("Gain", self)
        self.gain_label = widgets.QLabel("0", self)
        self.gain_control = widgets.QSlider(Qt.Vertical, self)
        self.gain_control.setTickPosition(widgets.QSlider.TicksBothSides)
        self.gain_control.setMinimum(-10)
        self.gain_control.setMaximum(20)
        self.gain_control.setValue(0)
        self.gain_control.setTickInterval(2)
        self.gain_control.setSingleStep(1)
        self.gain_control.setPageStep(2)

        self.threshold_title = widgets.QLabel("Threshold", self)
        self.threshold_label = widgets.QLabel(str(Settings.DEFAULT_POWER_THRESHOLD), self)
        self.slider = widgets.QSlider(Qt.Vertical, self)
        self.slider.setTickPosition(widgets.QSlider.TicksBothSides)
        self.slider.setMinimum(Settings.MIN_POWER_THRESHOLD)
        self.slider.setMaximum(Settings.MAX_POWER_THRESHOLD)
        self.slider.setValue(Settings.DEFAULT_POWER_THRESHOLD)
        self.slider.setTickInterval(20)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(20)

        self.gain_control.valueChanged.connect(self.on_gain_change)
        self.slider.valueChanged.connect(self.on_threshold_change)
        self.name_button.clicked.connect(self.run_name_change)

        layout = widgets.QGridLayout()
        layout.addWidget(self.name_button, 1, 1, 1, 2)
        layout.addWidget(self.gain_title, 2, 1)
        layout.addWidget(self.gain_label, 3, 1)
        layout.addWidget(self.gain_control, 4, 1)
        layout.addWidget(self.threshold_title, 2, 2)
        layout.addWidget(self.threshold_label, 3, 2)
        layout.addWidget(self.slider, 4, 2)
        self.setLayout(layout)

    @property
    def display_name(self):
        if self.name:
            return "{}: {}".format(self.idx, self.name)
        else:
            return "{}".format(self.idx)

    @property
    def short_name(self):
        if len(self.display_name) > 14:
            return self.display_name[:11] + "..."
        return self.display_name

    def on_gain_change(self, value):
        """TODO: make this gain per channel"""
        self.SET_GAIN.emit(value)
        self.gain_label.setText(str(value))

    def on_threshold_change(self, value):
        self.SET_THRESHOLD.emit(value)
        self.threshold_label.setText(str(value))

    def run_name_change(self, value):
        value, okay = widgets.QInputDialog.getText(
                self,
                "Bird/Channel Name",
                "Name",
                widgets.QLineEdit.Normal,
                self.name)
        if okay:
            self.name = value or None
            self.name_button.setText(self.short_name)
            self.name_button.setToolTip(self.display_name)
            self.SET_NAME.emit(self.name)


class RecordingWindow(widgets.QFrame):

    UPDATE_CHANNEL_NAMES = pyqtSignal(dict)

    def __init__(self, channels, parent=None):
        super(RecordingWindow, self).__init__(parent=parent)
        self.channels = channels
        self._plots_created = 0
        self.reset()

    def reset(self):
        self.spec_plots = {}
        self.controllers = {}
        self.channel_names = Settings.get("CHANNEL_NAMES", {})
        self.level_plots = {}
        self.curves = {}
        self.init_ui()

    def set_channels(self, channels):
        if channels != self.channels:
            self.channels = channels
            for ch_idx in range(self.channels):
                if ch_idx not in self.controllers:
                    self.init_channel(ch_idx)
            for ch_idx in range(self.channels):
                self.update_channel_names(
                    ch_idx,
                    self.channel_names.get(ch_idx)
                )

            for ch_idx in self.controllers:
                if ch_idx >= self.channels:
                    self.controllers[ch_idx].hide()
                    self.level_plots[ch_idx].hide()
                    self.spec_plots[ch_idx].hide()
                else:
                    self.controllers[ch_idx].show()
                    self.level_plots[ch_idx].show()
                    self.spec_plots[ch_idx].show()

    def init_channel(self, ch_idx):
        self.controllers[ch_idx] = RecordingController(
            idx=ch_idx,
            name=self.channel_names.get(ch_idx)
        )
        self.channel_names[ch_idx] = self.channel_names.get(ch_idx)
        self.level_plots[ch_idx] = WaveformWidget(
            Settings.CHUNK,
            show_x=False,
            window=Settings.PLOT_DURATION,
        )
        self.spec_plots[ch_idx] = SpectrogramWidget(
            Settings.CHUNK,
            min_freq=500,
            max_freq=10000,
            window=Settings.PLOT_DURATION,
            show_x=False, #True if ch_idx == self.channels - 1 else False,
            cmap=None
        )
        self.controllers[ch_idx].SET_THRESHOLD.connect(
            self.level_plots[ch_idx].set_threshold
        )
        self.controllers[ch_idx].SET_NAME.connect(
            partial(self.update_channel_names, ch_idx)
        )

        self.layout.setRowStretch(2 * ch_idx + 1, 3)
        self.layout.addWidget(
            self.spec_plots[ch_idx], 2 * ch_idx + 1, 2, 1, 1)
        self.layout.setRowStretch(2 * ch_idx + 2, 1)
        self.layout.addWidget(
            self.level_plots[ch_idx], 2 * ch_idx + 2, 2, 1, 1)

        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 10)
        self.layout.addWidget(
            self.controllers[ch_idx], 2 * ch_idx + 1, 1, 2, 1)

    def update_channel_names(self, ch_idx, name):
        self.channel_names[ch_idx] = name
        self.emit_channel_names()

    def emit_channel_names(self):
        self.UPDATE_CHANNEL_NAMES.emit({
            key: self.channel_names[key]
            for key in range(self.channels)
        })

    def init_ui(self):
        self.layout = widgets.QGridLayout()

        for ch_idx in range(self.channels):
            self.init_channel(ch_idx)

        self.setLayout(self.layout)

    def _loop(self):
        for ch_idx in range(self.channels):
            self.spec_plots[ch_idx].draw()
            self.level_plots[ch_idx].draw()
 
    @pyqtSlot(object)
    def receive_data(self, data):
        if data.shape[1] != self.channels:
            return

        for ch_idx in range(self.channels):
            self.spec_plots[ch_idx].receive_data(data[:, ch_idx])
            self.level_plots[ch_idx].receive_data(data[:, ch_idx])


def run(argv):
    app = widgets.QApplication(argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    run(sys.argv)
