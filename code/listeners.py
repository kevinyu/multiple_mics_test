import collections
import datetime
import logging

import os
import scipy.io.wavfile

import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QObject, QTimer

from settings import Settings
from utils import datetime2str
from ringbuffer import RingBuffer


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
    """Detects sound by looking at number of times the signal crosses a given threshold value
    """

    def __init__(self, size, parent=None):
        super(SoundDetector, self).__init__(parent)
        self._buffer = RingBuffer(maxlen=size)
        self._channels = None
        self.thresholds = {}

    def reset(self):
        self._buffer.clear()
        self._channels = None

    @pyqtSlot(int)
    def set_sampling_rate(self, sampling_rate):
        self._buffer.clear()
        self._buffer = RingBuffer(
            maxlen=int(Settings.DETECTION_BUFFER * sampling_rate)
        )

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
                np.diff(np.abs(dat[:, ch_idx]) > self.thresholds[ch_idx])
            )[0]

            ratio = int(threshold_crossings.size) / Settings.DETECTION_CROSSINGS_PER_CHUNK

            if ratio > 1:
                self.OUT.emit()
                break


class SoundSaver(MicrophoneListener):

    SAVE_EVENT = pyqtSignal(str)
    RECORDING = pyqtSignal(bool)

    def __init__(
            self,
            size,
            path,
            triggered=False,
            saving=False,
            channel_folders=None,
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
        self._buffer = RingBuffer()
        self._save_buffer = RingBuffer()
        self._idx = 0
        self.path = path
        self.saving = saving
        self.channel_folders = channel_folders
        self.triggered = triggered
        self.min_size = min_size
        self.sampling_rate = sampling_rate
        self.filename_format = filename_format
        self._file_idx = collections.defaultdict(int)
        self.size = size
        self._recording = False
        self._trigger_timer = None

        # self._trigger_timer.start(0.1)

    def start_rec(self):
        if self._trigger_timer:
            self._trigger_timer.stop()
            self._trigger_timer.deleteLater()
        else:
            self.RECORDING.emit(True)

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
        self.RECORDING.emit(False)
        self._recording = False
        self._trigger_timer = None

    @pyqtSlot(int)
    def set_sampling_rate(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self._buffer.clear()
        if self.triggered:
            self._buffer = RingBuffer(
                maxlen=int(Settings.DETECTION_BUFFER * self.sampling_rate)
            )
        else:
            self._buffer = RingBuffer()

    @pyqtSlot()
    def trigger(self):
        self.start_rec()

    def set_triggered(self, triggered):
        self.triggered = triggered
        self._buffer.clear()
        if self.triggered:
            self._buffer = RingBuffer(
                maxlen=int(Settings.DETECTION_BUFFER * self.sampling_rate)
            )
        else:
            self._buffer = RingBuffer()

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

        start_time = datetime.datetime.now() - datetime.timedelta(seconds=len(data) / Settings.RATE)
        time_str = datetime2str(start_time)

        if Settings.SAVE_CHANNELS_SEPARATELY and isinstance(self.channel_folders, list) and isinstance(self.filename_format, list):
            for channel, folder_name, filename_format in zip(range(self._channels), self.channel_folders, self.filename_format):
                folder_path = os.path.join(self.path, folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                print("Saving file to {}".format(folder_path))
                if Settings.FILENAME_SUFFIX == "time":
                    filename_str = filename_format.format(time_str)
                    path = os.path.join(folder_path, filename_str)
                else:
                    filename_str = filename_format.format(self._file_idx[channel])
                    path = os.path.join(folder_path, filename_str)
                    while os.path.exists(path):
                        self._file_idx[channel] += 1
                        filename_str = filename_format.format(self._file_idx[channel])
                        path = os.path.join(folder_path, filename_str)
                self.SAVE_EVENT.emit(path)
                scipy.io.wavfile.write(path, self.sampling_rate, data.astype(Settings.DTYPE)[:, channel])
        elif not Settings.SAVE_CHANNELS_SEPARATELY:
            folder_path = self.path
            filename_format = self.filename_format
            channel = None

            if Settings.FILENAME_SUFFIX == "time":
                filename_str = filename_format.format(time_str)
                path = os.path.join(folder_path, filename_str)
            else:
                filename_str = filename_format.format(self._file_idx[channel])
                path = os.path.join(folder_path, filename_str)
                while os.path.exists(path):
                    self._file_idx[channel] += 1
                    filename_str = filename_format.format(self._file_idx[channel])
                    path = os.path.join(folder_path, filename_str)

            self.SAVE_EVENT.emit(path)
            scipy.io.wavfile.write(path, self.sampling_rate, data.astype(Settings.DTYPE))
        else:
            raise Exception("When SAVE_CHANNELS_SEPARATELY is on, need channel_folders and filename_format to be lists")
