from __future__ import print_function
import os
import sys
import time
import collections
import threading
from queue import Queue
from functools import partial

import numpy as np
import pyaudio
import scipy.io.wavfile
import sounddevice

import sounddevice as sd


import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui as gui
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread, pyqtSlot, QSettings, Qt



from plotwidgets import SpectrogramWidget, WaveformWidget
from listeners import SoundDetector, SoundSaver, MicrophoneListener

from settings import Settings


class BaseMicrophone(QObject):
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

        # self.STOP.connect(self.stop)

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

    REC = pyqtSignal(object)
    STOP = pyqtSignal()

    def __init__(self, device_index=None, channels=1, parent=None):
        super(PyAudioMicrophone, self).__init__(device_index=device_index, channels=channels, parent=parent)
        self._stream = None
        self.p = pyaudio.PyAudio()
        self._stop = False

    def _run(self):
        def _callback(in_data, frame_count, time_info, status):
            if self._stop:
                return (None, pyaudio.paComplete)

            data = np.frombuffer(in_data, dtype=np.int16)[:]
            new_data = np.array([
                data[idx::self.channels]
                for idx in range(self.channels)
            ]).T

            self.REC.emit(new_data)

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

    REC = pyqtSignal(object)
    STOP = pyqtSignal()

    def __init__(self, device_index=None, channels=1, parent=None):
        super(SoundDeviceMicrophone, self).__init__(device_index=device_index, channels=channels, parent=parent)
        self._stream = None
        self.p = pyaudio.PyAudio()
        self._stop = False
        self._gain = Settings.GAIN

    def _run(self):
        def _callback(in_data, frame_count, time_info, status):
            scaled = np.power(10.0, Settings.GAIN / 20.0) * in_data
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
        self.bird_name = None

        self.init_ui()
        self.setup_listeners()
        self.connect_events()

        self.program_controller.save_location.setText(
            "[No path]/{0}".format(self.saver.filename_format)
        )

        self._get_selected_mic(None)
        self.set_channels(self.channels)
        self.connect_mic()

    def init_ui(self):
        main_frame = widgets.QFrame(self)
        self.layout = widgets.QVBoxLayout()

        self.scroll_area = widgets.QScrollArea(self)
        self.scroll_area.setFixedHeight(650)
        self.scroll_area.setFixedWidth(800)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.recording_window)

        self.layout.addWidget(self.program_controller)
        self.layout.addWidget(self.scroll_area)
        main_frame.setLayout(self.layout)

        self.setCentralWidget(main_frame)

    def setup_listeners(self):
        """Instaniate listeners
        """
        self.saver = SoundSaver(
            size=Settings.RATE * Settings.FILE_DURATION,
            min_size=Settings.RATE * Settings.MIN_FILE_DURATION,
            path=Settings.SAVE_DIRECTORY,
            triggered=False
        )
        self.saver.start()

        self.detector = SoundDetector(
            size=int(Settings.RATE * Settings.DETECTION_WINDOW)
        )
        self.detector.start()

        self.detector.OUT.connect(self.saver.trigger)

    def _get_selected_mic(self, idx):
        device = self.program_controller.input_source.currentData()
        self.on_select_mic(device.get("index"))

    def on_select_mic(self, device_index):
        self.channels = 1
        self.on_recording_mode("monitor")
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

        self.mic.set_channels(channels)

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

    def on_recording_mode(self, mode):
        if mode == "monitor":
            self.saver.set_triggered(False)
            self.saver.set_saving(False)
        elif mode == "triggered":
            self.saver.set_triggered(True)
            self.saver.set_saving(True)
        elif mode == "continuous":
            self.saver.set_triggered(False)
            self.saver.set_saving(True)

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
        self.program_controller.name_button.clicked.connect(self.run_bird_namer)

        self.program_controller.input_source.currentIndexChanged.connect(self._get_selected_mic)
        self.program_controller.input_source_channels.currentIndexChanged.connect(self.on_select_channels)

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
        else:
            display_path = os.path.join("[No path]", self.saver.filename_format)

        self.program_controller.save_location.setText(display_path)

    def run_bird_namer(self):
        value, okay = widgets.QInputDialog.getText(
                self,
                "Set Bird Name",
                "Bird Name",
                widgets.QLineEdit.Normal,
                self.bird_name)
        if okay:
            self.bird_name = value or None

        if self.bird_name:
            self.saver.filename_format = "{}_{{0}}.wav".format(self.bird_name)
        else:
            self.saver.filename_format = "recording_{0}.wav"
        
        self.update_display_path()

    def run_file_loader(self):
        options = widgets.QFileDialog.Options()
        path = widgets.QFileDialog.getExistingDirectory(
            self,
            "Save recordings to",
            self.saver.path or Settings.BASE_DIRECTORY,
            options=options)
        if path:
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
        self.name_button = widgets.QPushButton("Set bird name", self)
        self.save_location = widgets.QLabel("[No path]", self)

        self.monitor_button = widgets.QRadioButton("Monitor only")
        self.monitor_button.setChecked(True)
        self.trigger_button = widgets.QRadioButton(
                "Triggered recording")
        self.continuous_button = widgets.QRadioButton("Continuous recording")

        for i, device in enumerate(sd.query_devices()):
            if not device["max_input_channels"]:
                continue
            device["index"] = i
            self.input_source.addItem(device.get("name"), device)

        '''
        for i in range(self.p.get_device_count()):
            device = self.p.get_device_info_by_index(i)
            if not device.get("maxInputChannels"):
                continue
            self.input_source.addItem(device.get("name"), device)
        '''
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
        layout.addWidget(self.save_button, 1, 3)
        layout.addWidget(self.name_button, 2, 3)
        layout.addWidget(self.save_location, 3, 3)

        self.setLayout(layout)

    def update_channel_dropdown(self, idx):
        device = self.input_source.currentData()
        self.input_source_channels.clear()
        for i in range(device.get("max_input_channels")):
            self.input_source_channels.addItem(str(i + 1), i + 1)


class RecordingController(widgets.QFrame):

    SET_THRESHOLD = pyqtSignal(int)

    def __init__(self, parent=None):
        """
        Control settings for recording on this channel

        i.e. gain and threshold
        """
        super(RecordingController, self).__init__(parent)
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
        self.slider.setTickInterval(1000)
        self.slider.setSingleStep(50)
        self.slider.setPageStep(500)

        self.gain_control.valueChanged.connect(self.on_gain_change)
        self.slider.valueChanged.connect(self.on_threshold_change)

        layout = widgets.QGridLayout()
        layout.addWidget(self.gain_title, 1, 1)
        layout.addWidget(self.gain_label, 2, 1)
        layout.addWidget(self.gain_control, 3, 1)
        layout.addWidget(self.threshold_title, 1, 2)
        layout.addWidget(self.threshold_label, 2, 2)
        layout.addWidget(self.slider, 3, 2)
        self.setLayout(layout)

    def on_gain_change(self, value):
        """TODO: make this gain per channel"""
        Settings.GAIN = value
        self.gain_label.setText(str(value))

    def on_threshold_change(self, value):
        self.SET_THRESHOLD.emit(value)
        self.threshold_label.setText(str(value))


class RecordingWindow(widgets.QFrame):

    def __init__(self, channels, parent=None):
        super(RecordingWindow, self).__init__(parent=parent)
        self.channels = channels
        self._plots_created = 0
        self.reset()

    def reset(self):
        self.spec_plots = {}
        self.controllers = {}
        self.level_plots = {}
        self.curves = {}
        self.init_ui()

    def set_channels(self, channels):
        if channels != self.channels:
            self.channels = channels
            for ch_idx in range(self.channels):
                if ch_idx not in self.controllers:
                    self.init_channel(ch_idx)

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
        self.controllers[ch_idx] = RecordingController()
        self.level_plots[ch_idx] = WaveformWidget(
            Settings.CHUNK,
            show_x=False,
            window=Settings.PLOT_DURATION,
        )
        self.spec_plots[ch_idx] = SpectrogramWidget(
            Settings.CHUNK,
            min_freq=500,
            max_freq=8000,
            window=Settings.PLOT_DURATION,
            show_x=False, #True if ch_idx == self.channels - 1 else False,
            cmap=None
        )
        self.controllers[ch_idx].SET_THRESHOLD.connect(
            self.level_plots[ch_idx].set_threshold
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
