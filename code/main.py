from __future__ import print_function
import os
import sys
import time
import collections
import threading
from queue import Queue

import numpy as np
import pyaudio
import scipy.io.wavfile


import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui as gui
from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread, pyqtSlot, QSettings



from plotwidgets import SpectrogramWidget
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
    # kSTOP = pyqtSignal(object)

    def __init__(self, device=None, channels=1, parent=None):
        super(Microphone, self).__init__(parent)
        self.channels = channels
        # self._queue = Queue()
        self._stream = None
        self.p = pyaudio.PyAudio()
        # self.STOP.connect(self.stop)
        self._stop = False

    # @pyqtSlot()        
    def _run(self):
        def _callback(in_data, frame_count, time_info, status):
            if self._stop:
                return (None, pyaudio.paComplete)

            data = np.frombuffer(in_data, dtype=np.int16)[:]
            new_data = np.array([
                data[idx::self.channels]
                for idx in range(self.channels)
            ]).T

            # self._queue.put(new_data)
            self.REC.emit(new_data)

            return (in_data, pyaudio.paContinue)

        self._stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=Settings.RATE,
            frames_per_buffer=Settings.CHUNK,
            input=True,
            output=False,
            # input_device_index=0,
            stream_callback=_callback,
        )

    def run(self):
        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.start()
        self._run()

    # def stop(self):
    #     self._thread.terminate()


class MainWindow(widgets.QMainWindow):

    def __init__(self, show_channels=0):
        super(MainWindow, self).__init__()
        self.title = "Recorder"
        self.channels = show_channels
        self.settings = QSettings("Recorder", "Theunissen Lab")

        self.frame_timer = QTimer()
        self.frame_timer.start(1000 * 1.0/60.0)

        self.recording_window = RecordingWindow(self.channels, self)
        self.setCentralWidget(self.recording_window)

        self.frame_timer.timeout.connect(self.recording_window._loop)
        # self.frame_timer.timeout.connect(self._performance)

        self._last_t = time.time()

    def _performance(self):
        new_t = time.time()
        dt = new_t - self._last_t
        print("FPS: {:.2f}".format(1/dt), end="\r")
        self._last_t = new_t


class RecordingWindow(widgets.QFrame):

    def __init__(self, channels, parent=None):
        super(RecordingWindow, self).__init__(parent=parent)
        self.channels = channels
        self.spec_plots = {}
        self.init_ui()

    def closeEvent(self, event):
        pass

    def init_ui(self):
        for ch_idx in range(self.channels):
            # self.spec_plots[ch_idx] = pg.PlotWidget(self)
            self.spec_plots[ch_idx] = SpectrogramWidget(
                Settings.CHUNK,
                min_freq=500,
                max_freq=8000,
                window=5,
                show_x=True if ch_idx == self.channels - 1 else False,
                cmap=None
            )
            # self.panel[ch_idx] = Panel(self)

        layout = widgets.QGridLayout()
        for ch_idx in range(self.channels):
            layout.addWidget(self.spec_plots[ch_idx], ch_idx + 1, 1, 1, 1)

        self.setLayout(layout)

    def _loop(self):
        for ch_idx, plot in self.spec_plots.items():
            plot.show()

    # @pyqtSlot(object)
    def receive_data(self, data):
        for ch_idx in range(self.channels):
            self.spec_plots[ch_idx].receive_data(data[:, ch_idx])



def run(argv):
    app = widgets.QApplication(argv)

    mic = Microphone(channels=2)
    mic.run()

    window = MainWindow(show_channels=2)
    window.show()

    saver = SoundSaver(
        size=Settings.RATE * Settings.FILE_DURATION,
        min_size=Settings.RATE * Settings.MIN_FILE_DURATION,
        path="temp/painting",
        triggered=False
    )
    saver.start()

    mic.REC.connect(window.recording_window.receive_data)
    mic.REC.connect(saver.IN.emit)

    if True:
        detector = SoundDetector(int(Settings.RATE * Settings.DETECTION_WINDOW))
        detector.start()
        saver.set_triggered(True)
        mic.REC.connect(detector.IN.emit)
        detector.OUT.connect(saver.trigger)

    # mic.REC.connect(saver.receive_data)
    # saver = MicrophoneListener()
    # mic.REC.connect(saver.receive_data)

    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    run(sys.argv)
