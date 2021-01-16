import logging

import numpy as np
import sounddevice as sd

from configure_logging import handler
from events import Signal
from buffer import Buffer


DTYPE = np.int16
CHUNK = 1024

logger = logging.getLogger(__name__)
logger.addHandler(handler)


class Microphone(object):
    """Stream audio data to a signal"""

    REC = Signal()
    SETUP = Signal()

    def __init__(self, device_index=None, device_name=None):
        if device_index is None and device_name is None:
            raise ValueError("Device name or index must be supplied")
        elif device_index is not None:
            self.device_index = device_index
            self.device_name = self._device_index_to_name(device_index)
        elif device_name is not None:
            self.device_name = device_name
            self.device_index = self._device_name_to_index(device_name)
        self.n_channels = 1
        self._stream = None

    def _device_index_to_name(self, device_index):
        return sd.query_devices()[device_index]["name"]

    def _device_name_to_index(self, device_name):
        return sd._get_device_id(device_name, None)

    def set_channels(self, channels):
        """Set the device to listen on given number of channels

        If a list is given, will assume that the integers
        in the list are channel indices, and will default to the maximum
        value in the list + 1.
        """
        if isinstance(channels, list):
            n_channels = np.max(channels) + 1
        else:
            n_channels = channels
        self.n_channels = n_channels
        if self._stream:
            self._stream.close()
        self.run()

    def stop(self):
        if self._stream:
            self._stream.close()

    def _callback(self, in_data, frame_count, time_info, status):
        """Pass on data received on the stream to the REC signal
        """
        self.REC.emit(in_data.astype(DTYPE))
        return

    def run(self):
        """Start an audio input stream and emit metadata
        """
        device_info = sd.query_devices()[self.device_index]
        # Could the device name have changed by now?
        rate = int(device_info["default_samplerate"])
        self.SETUP.emit({
            "rate": rate,
            "device_name": self.device_name,
            "device_index": self.device_index,
            "n_channels": self.n_channels,
            "device_info": device_info,
        })

        try:
            self._stream = sd.InputStream(
                dtype=DTYPE,
                samplerate=rate,
                blocksize=CHUNK,
                device=self.device_index,
                channels=self.n_channels,
                callback=self._callback,
            )
        except:
            pass
        else:
            self._stream.start()


class SoundDetector(object):
    """Emits DETECTED signal when sound crosses a given amplitude threshold
    """

    DETECTED = Signal()

    def __init__(self, threshold=None, crossings_threshold=None):
        """Create a detector for threshold crossings
        """
        self._buffer = Buffer(maxlen=size)
        self.threshold = threshold
        self.crossings_threshold = crossings_threshold

    def reset(self):
        self._buffer.clear()

    def set_sampling_rate(self, sampling_rate):
        self._buffer.clear()
        self._buffer = Buffer(
            maxlen=int(Settings.DETECTION_BUFFER * sampling_rate)
        )

    def set_threshold(self, threshold):
        self.threshold = threshold

    def receive_data(data):
        self._buffer.extend(data)
        dat = np.array(self._buffer)
        if not len(dat):
            return

        threshold_crossings = np.nonzero(
            np.diff(np.abs(dat) > self.threshold)
        )[0]
        ratio = int(threshold_crossings.size) / self.crossings_threshold

        if ratio > 1:
            self.DETECTED.emit()


class SoundSaver(MicrophoneListener):

    SAVE_EVENT = Signal()
    RECORDING = Signal()

    def __init__(
            self,
            size,
            path,
            triggered=False,
            saving=False,
            channel_folders=None,
            min_size=None,
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

    def check_timer(self):
        """Prepare to stop recording
        """

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
