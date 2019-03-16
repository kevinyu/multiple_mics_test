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


import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui as gui
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread, pyqtSlot, QSettings, Qt



from plotwidgets import SpectrogramWidget, WaveformWidget
from listeners import SoundDetector, SoundSaver, MicrophoneListener

from settings import Settings


GUI_WIDTH = 640
GUI_HEIGHT = 480

# RATE = 30000
# jSILENT_BUFFER = 1.0
CROSSINGS_THRESHOLD = 0.2 * 200 # (heuristic: 200 crossings per second during sound)
AMP_THRESHOLD = 100
WHITE = (255,255,255) #RGB
BLACK = (0,0,0) #RGB



class Microphone(QObject):

    REC = pyqtSignal(object)

    def __init__(self, device=None, channels=1, parent=None):
        super(Microphone, self).__init__(parent)
        self.channels = channels
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

        self._stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=Settings.RATE,
            frames_per_buffer=Settings.CHUNK,
            input=True,
            output=False,
            input_device_index=0,
            stream_callback=_callback,
        )

    def run(self):
        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.start()
        self._run()


class MainWindow(widgets.QMainWindow):

    def __init__(self, show_channels=0):
        super(MainWindow, self).__init__()
        self.title = "Recorder"
        self.channels = show_channels

        self.frame_timer = QTimer()
        self.frame_timer.start(1000 * 1.0/120.0)

        self.recording_window = RecordingWindow(self.channels, self)
        self.program_controller = ProgramController(self)

        main_frame = widgets.QFrame(self)
        layout = widgets.QVBoxLayout()
        layout.addWidget(self.program_controller)
        layout.addWidget(self.recording_window)
        main_frame.setLayout(layout)

        self.setCentralWidget(main_frame)

        self.frame_timer.timeout.connect(self.recording_window._loop)

        self._last_t = time.time()


class ProgramController(widgets.QFrame):

    def __init__(self, parent=None):
        super(ProgramController, self).__init__(parent=parent)
        self.p = pyaudio.PyAudio()

        self.init_ui()

    def init_ui(self):
        self.input_source = widgets.QComboBox(self)
        self.input_source_channels = widgets.QComboBox(self)
        self.save_button = widgets.QPushButton("Select save location", self)

        self.monitor_button = widgets.QRadioButton("Monitor only")
        self.trigger_button = widgets.QRadioButton(
                "Triggered recording")
        self.continuous_button = widgets.QRadioButton("Continuous recording")

        for i in range(self.p.get_device_count()):
            device = self.p.get_device_info_by_index(i)
            if not device.get("maxInputChannels"):
                continue
            self.input_source.addItem(device.get("name"), device)


        layout = widgets.QGridLayout()

        layout.addWidget(self.input_source, 1, 2)
        layout.addWidget(self.save_button, 1, 3)
        layout.addWidget(self.monitor_button, 1, 1)
        layout.addWidget(self.trigger_button, 2, 1)
        layout.addWidget(self.continuous_button, 3, 1)
        layout.addWidget(self.input_source_channels, 2, 2)

        self.save_button.clicked.connect(self.run_file_loader)

        self.setLayout(layout)

    def run_file_loader(self):
        options = widgets.QFileDialog.Options()
        path = widgets.QFileDialog.getExistingDirectory(
            self,
            "Save recordings to",
            Settings.BASE_DIRECTORY,
            options=options)

        # TODO: set save directory here


class RecordingController(widgets.QFrame):

    SET_THRESHOLD = pyqtSignal(int)

    def __init__(self, parent=None):
        """
        Control settings for recording on this channel

        i.e. gain and threshold
        """
        super(RecordingController, self).__init__(parent)
        self.slider = widgets.QSlider(Qt.Vertical)
        self.slider.setTickPosition(widgets.QSlider.TicksBothSides)
        self.slider.setMinimum(Settings.MIN_POWER_THRESHOLD)
        self.slider.setMaximum(Settings.MAX_POWER_THRESHOLD)
        self.slider.setValue(Settings.DEFAULT_POWER_THRESHOLD)
        self.slider.setTickInterval(50)
        self.slider.setSingleStep(10)
        self.slider.valueChanged.connect(self.SET_THRESHOLD.emit)

        layout = widgets.QVBoxLayout()
        layout.addWidget(self.slider)
        self.setLayout(layout)


class RecordingWindow(widgets.QFrame):

    def __init__(self, channels, parent=None):
        super(RecordingWindow, self).__init__(parent=parent)
        self.channels = channels
        self.spec_plots = {}
        self.controllers = {}
        self.level_plots = {}
        self.curves = {}
        self.init_ui()

    def closeEvent(self, event):
        pass

    def init_ui(self):
        for ch_idx in range(self.channels):
            self.controllers[ch_idx] = RecordingController()
            self.level_plots[ch_idx] = WaveformWidget(
                Settings.CHUNK,
                show_x=False,
                window=Settings.PLOT_DURATION,
            )
            self.spec_plots[ch_idx] = SpectrogramWidget(
                Settings.CHUNK,
                min_freq=500,
                max_freq=12000,
                window=Settings.PLOT_DURATION,
                show_x=False, #True if ch_idx == self.channels - 1 else False,
                cmap=None
            )
            self.controllers[ch_idx].SET_THRESHOLD.connect(
                self.level_plots[ch_idx].set_threshold
            )
            # self.panel[ch_idx] = Panel(self)

        layout = widgets.QGridLayout()
        for ch_idx in range(self.channels):
            layout.setRowStretch(2 * ch_idx + 1, 3)
            layout.addWidget(
                self.spec_plots[ch_idx], 2 * ch_idx + 1, 2, 1, 1)
            layout.setRowStretch(2 * ch_idx + 2, 1)
            layout.addWidget(
                self.level_plots[ch_idx], 2 * ch_idx + 2, 2, 1, 1)

            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 10)
            layout.addWidget(
                self.controllers[ch_idx], 2 * ch_idx + 1, 1, 2, 1)

        self.setLayout(layout)

    def _loop(self):
        for ch_idx in range(self.channels):
            self.spec_plots[ch_idx].show()
            self.level_plots[ch_idx].show()
 
    @pyqtSlot(object)
    def receive_data(self, data):
        for ch_idx in range(self.channels):
            self.spec_plots[ch_idx].receive_data(data[:, ch_idx])
            self.level_plots[ch_idx].receive_data(data[:, ch_idx])


def run(argv):
    app = widgets.QApplication(argv)

    saver = SoundSaver(
        size=Settings.RATE * Settings.FILE_DURATION,
        min_size=Settings.RATE * Settings.MIN_FILE_DURATION,
        path=Settings.SAVE_DIRECTORY,
        triggered=False
    )
    saver.start()

    mic = Microphone(channels=2)
    mic.run()

    window = MainWindow(show_channels=2)
    window.show()

    mic.REC.connect(window.recording_window.receive_data)

    if not Settings.SAVE_CONTINUOUSLY:
        detector = SoundDetector(
            size=int(Settings.RATE * Settings.DETECTION_WINDOW)
        )
        detector.start()
        saver.set_triggered(True)
        mic.REC.connect(detector.IN.emit)
        detector.OUT.connect(saver.trigger)

        for ch_idx in range(2):
            (window.recording_window
                .controllers[ch_idx]
                .SET_THRESHOLD
                .connect(partial(detector.set_threshold, ch_idx)))
        # detector.LEVEL.connect(window.recording_window.update_crossings)

    mic.REC.connect(saver.receive_data)
    # saver = MicrophoneListener()
    # mic.REC.connect(saver.receive_data)

    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    run(sys.argv)
