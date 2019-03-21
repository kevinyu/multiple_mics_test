import collections
import logging
import time

import os
import scipy.io.wavfile

import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QObject, QTimer

from settings import Settings
import itertools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MicrophoneListener(QObject):

    IN = pyqtSignal(object)
    OUT = pyqtSignal()

    @pyqtSlot(object)
    def receive_data(self, data):
        logger.info("{} received data of size {}".format(
            self, data.shape))

    def start(self):
        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.start()
        self.IN.connect(self.receive_data)

    def stop(self):
        self._thread.terminate()


class SoundDetector(MicrophoneListener):

    def __init__(self, size, parent=None):
        super(SoundDetector, self).__init__(parent)
        self._buffer = collections.deque(maxlen=size)
        self._channels = None
        self.thresholds = {}

    def reset(self):
        self._buffer.clear()
        self._channels = None

    def set_threshold(self, ch, threshold):
        self.thresholds[ch] = threshold

    @pyqtSlot(object)
    def receive_data(self, data):
        if self._channels is None:
            self._channels = data.shape[1]

        if data.shape[1] != self._channels:
            return

        self._buffer.extend(data)

        dat = np.array(self._buffer)

        if not len(dat):
            return

        for ch_idx in range(dat.shape[1]):
            if ch_idx not in self.thresholds:
                self.thresholds[ch_idx] = Settings.DEFAULT_POWER_THRESHOLD
            threshold_crossings = np.nonzero(
                np.diff(np.power(np.abs(dat[:, ch_idx]), 2) > self.thresholds[ch_idx])
            )[0]

            ratio = int(threshold_crossings.size) / Settings.DETECTION_CROSSINGS_PER_CHUNK

            if ratio > 1:
                print("ON    ", end="\r")
                self.OUT.emit()
                break
        else:
            print("OFF   ", end="\r")


class SoundSaver(MicrophoneListener):

    def __init__(
            self,
            size,
            path,
            triggered=False,
            saving=False,
            filename_format="recording_{0}.wav",
            min_size=None,
            sampling_rate=44100,
            parent=None
        ):
        """
        Parameters
        ----------
        size : int
            Size of each file to be saved in samples
        """
        super(SoundSaver, self).__init__(parent)
        self._buffer = collections.deque()
        self._save_buffer = collections.deque()
        self._idx = 0
        self.path = path
        self.saving = saving
        self.triggered = triggered
        self.min_size = min_size
        self.sampling_rate = sampling_rate
        self.filename_format = filename_format
        self._file_idx = 0
        self.size = size
        self._recording = False
        self._trigger_timer = None

        # self._trigger_timer.start(0.1)

    def start_rec(self):
        if self._trigger_timer:
            self._trigger_timer.stop()
            self._trigger_timer.deleteLater()

        self._trigger_timer = QTimer(self)
        self._trigger_timer.timeout.connect(self.stop_rec)
        self._trigger_timer.setSingleShot(True)
        self._trigger_timer.start(Settings.DETECTION_BUFFER * 1000)
        self._recording = True
        self._channels = None

    def reset(self):
        self._buffer.clear()
        self._save_buffer.clear()
        self._channels = None

    def stop_rec(self):
        self._recording = False

    @pyqtSlot()
    def trigger(self):
        self.start_rec()

    def set_triggered(self, triggered):
        self.triggered = triggered
        self._buffer.clear()
        if self.triggered:
            self._buffer = collections.deque(
                maxlen=int(Settings.DETECTION_BUFFER * Settings.RATE)
            )
        else:
            self._buffer = collections.deque()

    def set_saving(self, saving):
        self.saving = saving

    @pyqtSlot(object)
    def receive_data(self, data):
        if self._channels is None:
            self._channels = data.shape[1]

        if data.shape[1] != self._channels:
            self._buffer.clear()
            self._save_buffer.clear()
            return

        if not self.saving:
            self._buffer.clear()
            self._save_buffer.clear()
            return

        self._buffer.extend(data)

        if not self.triggered:
            if len(self._buffer) > self.size:
                data = np.array(self._buffer)
                self._save(data[:self.size])
                self._buffer.clear()
                self._buffer.extend(data[self.size:])
        if self.triggered:
            if self._recording:
                if not len(self._save_buffer):
                    self._save_buffer.extend(self._buffer)
                else:
                    self._save_buffer.extend(data)

                if (len(self._save_buffer) / Settings.RATE) >= Settings.MAX_TRIGGERED_DURATION:
                    data = np.array(self._save_buffer)
                    self._save(data)
                    self._save_buffer.clear()
            else:
                data_to_save = np.array(self._save_buffer)
                if not self.min_size or len(data_to_save) > self.min_size:
                    self._save(data_to_save)
                self._save_buffer.clear()

    def _save(self, data):
        if not self.saving:
            return

        if not self.path:
            print("Warning: No path is configured")
            return 

        if not os.path.exists(self.path):
            print("Warning: {} does not exist".format(self.path))
            return

        filename = self.filename_format.format(self._file_idx)
        path = os.path.join(self.path, filename)
        while os.path.exists(path):
            self._file_idx += 1
            filename = self.filename_format.format(self._file_idx)
            path = os.path.join(self.path, filename)

        print("\nSaving file to {}\n".format(path))
        scipy.io.wavfile.write(path, self.sampling_rate, data.astype(np.int16))
