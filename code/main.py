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
from PyQt5.QtCore import QTimer, pyqtSignal, QObject


CHUNK = 512 * 1
WIDTH = 2
GUI_WIDTH = 640
GUI_HEIGHT = 480

RATE = 30000
SILENT_BUFFER = 1.0
CROSSINGS_THRESHOLD = 0.2 * 200 # (heuristic: 200 crossings per second during sound)
AMP_THRESHOLD = 100
WHITE = (255,255,255) #RGB
BLACK = (0,0,0) #RGB



class Microphone(QObject):

    REC = pyqtSignal(object)
    STOP = pyqtSignal(object)

    def __init__(self, device=None, channels=1):
        super(Microphone, self).__init__()
        self.channels = channels
        self._queue = Queue()
        self._stream = None
        self.p = pyaudio.PyAudio()
        self.STOP.connect(self.stop)
        self._stop = False

    def _run(self, queue):
        def _callback(in_data, frame_count, time_info, status):
            if self._stop:
                return (None, pyaudio.paComplete)

            data = np.frombuffer(in_data, dtype=np.int16)[:]
            new_data = np.array([
                data[idx::self.channels]
                for idx in range(self.channels)
            ]).T

            queue.put(new_data)
            self.REC.emit(new_data)

            return (in_data, pyaudio.paContinue)

        self._stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=RATE,
            frames_per_buffer=CHUNK,
            input=True,
            output=False,
            input_device_index=7,
            stream_callback=_callback,
        )

    def run(self):
        self._thread = threading.Thread(target=self._run, args=(self._queue,))
        self._thread.run()

    def stop(self):
        self._stop = True
        self._thread.join()


class SpectrogramWidget(pg.PlotWidget):
    """Live spectrogram widgets

    Based off code from here:
    http://amyboyle.ninja/Pyqtgraph-live-spectrogram
    """

    def __init__(
            self,
            chunk_size,
            min_freq=500,
            max_freq=8000,
            window=5,
            cmap=None
        ):

        super(SpectrogramWidget, self).__init__()
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.chunk_size = chunk_size
        freq = (
            np.arange((self.chunk_size // 2) + 1) /
            (float(self.chunk_size) / RATE)
        )
        self.freq_mask = np.where(
            (min_freq < freq) &
            (freq < max_freq)
        )[0]

        self.img_array = np.zeros((
            window * int(RATE / CHUNK),
            len(freq[self.freq_mask])
        ))
        self.cmap = self.get_cmap() if cmap is None else cmap

        lut = self.cmap.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        yscale = (
            1.0 /
            (self.img_array.shape[1] / freq[self.freq_mask][-1])
        )

        self.img.scale((1. / RATE) * self.chunk_size, yscale)
        self.win = np.hanning(self.chunk_size)

    def get_cmap(self):
        pos = np.array([0.1, 1])
        color = np.array([
            (255,255,255,255),
            (0, 0, 0, 255)
        ], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)

        return cmap

    def push_data(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(chunk * self.win) / self.chunk_size
        psd = abs(spec) ** 2           # magnitude
        psd = 20 * np.log10(psd)       # to dB scale

        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = psd[self.freq_mask]

    def show(self):
        self.img.setImage(self.img_array, autoLevels=False)


class MainWindow(widgets.QMainWindow):

    def __init__(self, show_channels=0):
        super(MainWindow, self).__init__()
        self.title = "Recorder"
        self.channels = show_channels

        self.frame_timer = QTimer()
        self.frame_timer.start(1.0/30.0)

        self.recording_window = RecordingWindow(self.channels, self)
        self.setCentralWidget(self.recording_window)

        self.frame_timer.timeout.connect(self.recording_window._loop)
        # self.frame_timer.timeout.connect(self._performance)

        self._last_t = time.time()

    def _performance(self):
        new_t = time.time()
        dt = new_t - self._last_t
        print("FPS: {:.2f}".format(1/dt))
        self._last_t = new_t


class RecordingWindow(widgets.QFrame):

    def __init__(self, channels, parent=None):
        super(RecordingWindow, self).__init__(parent=parent)
        self.channels = channels
        self.spec_plots = {}
        self.init_ui()

    def init_ui(self):
        for ch_idx in range(self.channels):
            # self.spec_plots[ch_idx] = pg.PlotWidget(self)
            self.spec_plots[ch_idx] = SpectrogramWidget(
                CHUNK,
                min_freq=500,
                max_freq=8000,
                window=5,
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

    def receive_data(self, data):
        for ch_idx in range(self.channels):
            self.spec_plots[ch_idx].push_data(data[:, ch_idx])



def run(argv):
    mic = Microphone(channels=2)
    mic.run()

    app = widgets.QApplication(argv)

    window = MainWindow(show_channels=2)
    window.show()
    mic.REC.connect(window.recording_window.receive_data)

    # detector = SoundDetector()
    # mic.REC.connect(detector.receive_data)

    # saver = SoundSaver(RATE * 10)
    # mic.REC.connect(saver.receive_data)

    sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    run(sys.argv)
