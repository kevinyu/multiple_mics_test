import asyncio
import logging
from functools import partial

import numpy as np
import sounddevice as sd

from configure_logging import handler
from events import Signal
from buffer import Buffer
from utils import db_scale


DTYPE = np.int16
CHUNK = 1024
DETECTION_BUFFER = 0.3


logger = logging.getLogger(__name__)
logger.addHandler(handler)


class Filter(object):
    """Filter(s) applied to a microphone object to process data received
    """
    def apply(self, data):
        return data


class GainFilter(Filter):
    def __init__(self, channel_gains: dict):
        self.channel_gains = channel_gains

    def apply(self, data):
        for i in self.channel_gains:
            if i < data.shape[1]:
                data[:, i] = db_scale(data[:, i], self.channel_gains[i])
        return data


class FilteredMicrophone(object):
    """A dummy microphone with a REC signal for processed output

    Separates the mic thread from the main app thread
    """
    def __init__(self, mic, *filters):
        self.mic = mic
        self.filters = filters
        self.mic.REC.connect(self.receive_data)

        self.REC = Signal()
        self.SETUP = self.mic.SETUP

    def receive_data(self, data):
        for filter in self.filters:
            data = filter.apply(data)
        self.REC.emit(data)


class Microphone(object):
    """Stream audio data to a signal"""

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
        self.mic_queue = asyncio.Queue()

        self.REC = Signal()
        self.SETUP = Signal()

    def apply(self, *filters):
        for filter in filters:
            if not isinstance(filter, Filter):
                raise ValueError("Can only apply Filters to mic")

        return FilteredMicrophone(self, *filters)

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

    def stop(self):
        if self._stream:
            self._stream.close()

    def _callback(self, loop, in_data, frame_count, time_info, status):
        """Pass on data received on the stream to the REC signal
        """
        loop.call_soon_threadsafe(
            self.mic_queue.put_nowait,
            in_data.astype(DTYPE)
        )
        return

    async def run(self):
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

        loop = asyncio.get_event_loop()

        try:
            self._stream = sd.InputStream(
                dtype=DTYPE,
                samplerate=rate,
                blocksize=CHUNK,
                device=self.device_index,
                channels=self.n_channels,
                callback=partial(self._callback, loop),
            )
        except:
            raise
        else:
            self._stream.start()

        running = True
        while running:
            try:
                chunk = await self.mic_queue.get()
            except:
                running = False
                raise
            else:
                self.REC.emit(chunk)


class SoundDetector(object):
    """Emits DETECTED signal when sound crosses a given amplitude threshold

    Emits the channel indexes that the sound was detected on
    """

    def __init__(
            self,
            detection_window: float,
            default_threshold: float,
            crossings_threshold: int,
            ):
        """Create a detector for threshold crossings

        Parameters
        ==========
        threshold : float
            Amplitude threshold
        crossings_threshold : float
            Emit detection signal when the number of threshold crossings
            surpasses this value
        detection_window : float
            Window size in seconds to count threshold crossings in
        """
        self._buffer = Buffer(maxlen=0)
        self.detection_window = detection_window
        self.default_threshold = default_threshold
        self.crossings_threshold = crossings_threshold

        self.reset()

        self.DETECTED = Signal()

    def reset(self):
        self._buffer.clear()
        self.thresholds = {}

    def set_sampling_rate(self, sampling_rate):
        self._buffer.clear()
        self._buffer = Buffer(
            maxlen=int(self.detection_window * sampling_rate)
        )

    def set_channel_threshold(self, channel, threshold):
        self.thresholds[channel] = threshold

    def receive_data(self, data):
        if self._buffer.maxlen == 0:
            return

        self._buffer.extend(data)
        dat = np.array(self._buffer)

        if not len(dat):
            return

        for ch_idx in range(dat.shape[1]):
            if ch_idx not in self.thresholds:
                self.thresholds[ch_idx] = self.default_threshold

            threshold_crossings = np.nonzero(
                np.diff(np.abs(dat[:, ch_idx]) > self.thresholds[ch_idx])
            )[0]

            ratio = int(threshold_crossings.size) / self.crossings_threshold
            # print(ratio)
            if ratio > 1:
                self.DETECTED.emit(ch_idx)


###
#



class SoundSaver(object):
    """
    """
    DEFAULT_MIN_FILE_DURATION = 1.0
    DEFAULT_MAX_FILE_DURATION = 30.0
    TRIGGER_DECAY_TIME = 0.2

    def __init__(
            self,
            sampling_rate=None,
            min_file_duration=DEFAULT_MIN_FILE_DURATION,
            max_file_duration=DEFAULT_MAX_FILE_DURATION,
            ):
        """

        max_file_duration : int
            Maximum file length in seocnds. Provides a safeguard to
            triggers set too low where saving is constantly on. Acutal
            file size depends on the sampling rate.
        """
        self.status = {
            "recording": False,
            "ready": False,
        }

        self._buffer = None
        self._save_buffer = Buffer()
        self._trigger_timer = Buffer()

        self.min_file_duration = min_file_duration
        self.max_file_duration = max_file_duration
        self.set_sampling_rate(sampling_rate)

        self.SAVE_READY = Signal()
        self.RECORDING = Signal()

    def set_sampling_rate(self, sampling_rate):
        if sampling_rate is None:
            self.sampling_rate = None
            self.min_file_length = None
            self.max_file_length = None
            self.status["ready"] = False
        else:
            self.sampling_rate = sampling_rate
            self.min_file_length = int(self.min_file_duration * self.sampling_rate)
            self.max_file_length = int(self.max_file_duration * self.sampling_rate)
            self.status["ready"] = True
            self._buffer = Buffer(maxlen=int(self.TRIGGER_DECAY_TIME * self.sampling_rate))
            self._save_buffer = Buffer(maxlen=int(self.max_file_length))

    def trigger(self):
        """Initiates the saving of a file"""
        loop = asyncio.get_running_loop()

        if self._trigger_timer:
            self._trigger_timer.cancel()
            self._trigger_timer = None
        else:
            self.RECORDING.emit(True)

        self._trigger_timer = loop.call_later(
            self.TRIGGER_DECAY_TIME,
            self.stop_recording,
        )
        self.status["recording"] = True

    def stop_recording(self):
        self.status["recording"] = False
        self._trigger_timer = None
        self.RECORDING.emit(False)

    def receive_data(self, data):
        if not self.status["ready"]:
            self._buffer.clear()
            self._save_buffer.clear()
            return

        self._buffer.extend(data)

        if self.status["recording"]:
            if not len(self._save_buffer):
                self._save_buffer.extend(np.array(self._buffer))
            else:
                self._save_buffer.extend(data)

            if (len(self._save_buffer) / self.sampling_rate) >= self.max_file_length:
                data = np.array(self._save_buffer)
                self.SAVE_READY.emit(data)
                self._save_buffer.clear()
        else:
            data_to_save = np.array(self._save_buffer)
            if not self.min_file_length or len(data_to_save) > self.min_file_length:
                self.SAVE_READY.emit(data_to_save)
            self._save_buffer.clear()


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
