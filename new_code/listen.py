import asyncio
import logging
from functools import partial

import numpy as np
import sounddevice as sd

from configure_logging import handler
from events import Signal
from ringbuffer import RingBuffer
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
    """A filter that applies gain in dB per channel

    Config
    ======
    channel_gains: dict
        Dict mapping channel index to gain value
    """
    def __init__(self, channel_gains: dict):
        self.channel_gains = channel_gains

    def configure(self, config):
        pass

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
    """Stream audio data to a signal

    Config
    ======
    device_index: int
    device_name: str
    channels: int
    """

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


class StreamManager(object):
    """Manages a stream"""
    def __init__(self, mic, detector):
        self.mic = mic
        self.detector = detector

    def apply_config(self, config):
        # Apply proper gain to filter
        # Hook up a saver
        # Set up detector params
        # 
        pass
        


class SoundDetector(object):
    """Emits DETECTED signal when sound crosses a given amplitude threshold

    Emits the channel indexes that the sound was detected on

    Config
    ======
    thresholds: dict
        Map channel index to detection threshold. Channels not defined
        in this mapping will use a default threshold value.
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
        crossings_threshold : float
            Emit detection signal when the number of threshold crossings
            surpasses this value
        default_threshold : float
            Default amplitude threshold value to use when threshold for
            a channel is not explicitly set
        detection_window : float
            Window size in seconds to count threshold crossings in
        """
        self._buffer = RingBuffer()
        self.detection_window = detection_window
        self.default_threshold = default_threshold
        self.crossings_threshold = crossings_threshold

        self.reset()

        self.DETECTED = Signal()

    def reset(self):
        self._buffer.clear()
        # Thresholds will be set to default the first time they are accessed
        self.thresholds = {}

    def set_sampling_rate(self, sampling_rate):
        self._buffer = RingBuffer(maxlen=int(self.detection_window * sampling_rate))

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
            # Thresholds get set here if they are missing
            if ch_idx not in self.thresholds:
                self.thresholds[ch_idx] = self.default_threshold

            threshold_crossings = np.nonzero(
                np.diff(np.abs(dat[:, ch_idx]) > self.thresholds[ch_idx])
            )[0]

            ratio = int(threshold_crossings.size) / self.crossings_threshold
            # print(ratio)
            if ratio > 1:
                self.DETECTED.emit(ch_idx)


class SoundSaver(object):
    """A triggered sound saver
    """
    DEFAULT_MIN_FILE_DURATION = 1.0
    DEFAULT_MAX_FILE_DURATION = 30.0
    DEFAULT_BUFFER_DURATION = 0.2

    def __init__(
            self,
            sampling_rate=None,
            min_file_duration=DEFAULT_MIN_FILE_DURATION,
            max_file_duration=DEFAULT_MAX_FILE_DURATION,
            buffer_duration=DEFAULT_BUFFER_DURATION,
            ):
        """Initialize a SoundSaver instance and buffers

        Params
        ======
        sampling_rate : int
            Sampling rate - if None SoundSaver instance will be inactive.
            Update with SoundSaver.set_sampling_rate()
        max_file_duration : float
            Maximum file length in seconds. Provides a safeguard to
            triggers set too low where saving is constantly on. Acutal
            file size depends on the sampling rate.
        min_file_duration : float
            Minimum file length in seconds. Do not emit save events
            for sound periods shorter than this length
        buffer_duration : float (default = 0.2)
            How many seconds before and after the triggered period
            to capture. This determines the buffer length as well as
            the amount of silence needed to turn a triggered recording off.
            Each call to trigger() resets this timer.
        """
        self.status = {
            "recording": False,
            "ready": False,
        }

        self._buffer = RingBuffer()
        self._save_buffer = RingBuffer()
        self._trigger_timer = None

        self.min_file_duration = min_file_duration
        self.max_file_duration = max_file_duration
        self.buffer_duration = buffer_duration
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
            self._buffer = RingBuffer(maxlen=int(self.buffer_duration * self.sampling_rate))
            self._save_buffer = RingBuffer(maxlen=int(self.max_file_length))

    def trigger(self):
        """Initiates the saving of a file"""
        loop = asyncio.get_running_loop()

        if self._trigger_timer:
            self._trigger_timer.cancel()
            self._trigger_timer = None
        else:
            self.RECORDING.emit(True)

        self._trigger_timer = loop.call_later(
            self.buffer_duration,
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



#
# class SoundSaver(MicrophoneListener):
#
#     SAVE_EVENT = Signal()
#     RECORDING = Signal()
#
#     def __init__(
#             self,
#             size,
#             path,
#             triggered=False,
#             saving=False,
#             channel_folders=None,
#             min_size=None,
#             parent=None
#             ):
#         """
#         Parameters
#         ----------
#         size : int
#             Size of each file to be saved in samples
#         """
#         super(SoundSaver, self).__init__(parent)
#         self._buffer = RingBuffer()
#         self._save_buffer = RingBuffer()
#         self._idx = 0
#         self.path = path
#         self.saving = saving
#         self.channel_folders = channel_folders
#         self.triggered = triggered
#         self.min_size = min_size
#         self.sampling_rate = sampling_rate
#         self.filename_format = filename_format
#         self._file_idx = collections.defaultdict(int)
#         self.size = size
#         self._recording = False
#         self._trigger_timer = None
#
#         # self._trigger_timer.start(0.1)
#
#     def check_timer(self):
#         """Prepare to stop recording
#         """
#
#     def start_rec(self):
#         if self._trigger_timer:
#             self._trigger_timer.stop()
#             self._trigger_timer.deleteLater()
#         else:
#             self.RECORDING.emit(True)
#
#         self._trigger_timer = QTimer(self)
#         self._trigger_timer.timeout.connect(self.stop_rec)
#         self._trigger_timer.setSingleShot(True)
#         self._trigger_timer.start(Settings.DETECTION_BUFFER * 1000)
#         self._recording = True
#         self._channels = None
#
#     def reset(self):
#         self._buffer.clear()
#         self._save_buffer.clear()
#         self._channels = None
#
#     def stop_rec(self):
#         self.RECORDING.emit(False)
#         self._recording = False
#         self._trigger_timer = None
#
#     @pyqtSlot(int)
#     def set_sampling_rate(self, sampling_rate):
#         self.sampling_rate = sampling_rate
#         self._buffer.clear()
#         if self.triggered:
#             self._buffer = RingBuffer(
#                 maxlen=int(Settings.DETECTION_BUFFER * self.sampling_rate)
#             )
#         else:
#             self._buffer = RingBuffer()
#
#     @pyqtSlot()
#     def trigger(self):
#         self.start_rec()
#
#     def set_triggered(self, triggered):
#         self.triggered = triggered
#         self._buffer.clear()
#         if self.triggered:
#             self._buffer = RingBuffer(
#                 maxlen=int(Settings.DETECTION_BUFFER * self.sampling_rate)
#             )
#         else:
#             self._buffer = RingBuffer()
#
#     def set_saving(self, saving):
#         self.saving = saving
#
#     @pyqtSlot(object)
#     def receive_data(self, data):
#         if self._channels is None:
#             self._channels = data.shape[1]
#
#         if data.shape[1] != self._channels:
#             self._buffer.clear()
#             self._save_buffer.clear()
#             return
#
#         if not self.saving:
#             self._buffer.clear()
#             self._save_buffer.clear()
#             return
#
#         self._buffer.extend(data)
#
#         if not self.triggered:
#             if len(self._buffer) > self.size:
#                 data = np.array(self._buffer)
#                 self._save(data[:self.size])
#                 self._buffer.clear()
#                 self._buffer.extend(data[self.size:])
#
#         if self.triggered:
#             if self._recording:
#                 if not len(self._save_buffer):
#                     self._save_buffer.extend(self._buffer)
#                 else:
#                     self._save_buffer.extend(data)
#
#                 if (len(self._save_buffer) / Settings.RATE) >= Settings.MAX_TRIGGERED_DURATION:
#                     data = np.array(self._save_buffer)
#                     self._save(data)
#                     self._save_buffer.clear()
#             else:
#                 data_to_save = np.array(self._save_buffer)
#                 if not self.min_size or len(data_to_save) > self.min_size:
#                     self._save(data_to_save)
#                 self._save_buffer.clear()
#
#     def _save(self, data):
#         if not self.saving:
#             return
#
#         if not self.path:
#             print("Warning: No path is configured")
#             return
#
#         if not os.path.exists(self.path):
#             print("Warning: {} does not exist".format(self.path))
#             return
#
#         start_time = datetime.datetime.now() - datetime.timedelta(seconds=len(data) / Settings.RATE)
#         time_str = datetime2str(start_time)
#
#         if Settings.SAVE_CHANNELS_SEPARATELY and isinstance(self.channel_folders, list) and isinstance(self.filename_format, list):
#             for channel, folder_name, filename_format in zip(range(self._channels), self.channel_folders, self.filename_format):
#                 folder_path = os.path.join(self.path, folder_name)
#                 if not os.path.exists(folder_path):
#                     os.makedirs(folder_path)
#                 print("Saving file to {}".format(folder_path))
#                 if Settings.FILENAME_SUFFIX == "time":
#                     filename_str = filename_format.format(time_str)
#                     path = os.path.join(folder_path, filename_str)
#                 else:
#                     filename_str = filename_format.format(self._file_idx[channel])
#                     path = os.path.join(folder_path, filename_str)
#                     while os.path.exists(path):
#                         self._file_idx[channel] += 1
#                         filename_str = filename_format.format(self._file_idx[channel])
#                         path = os.path.join(folder_path, filename_str)
#                 self.SAVE_EVENT.emit(path)
#                 scipy.io.wavfile.write(path, self.sampling_rate, data.astype(Settings.DTYPE)[:, channel])
#         elif not Settings.SAVE_CHANNELS_SEPARATELY:
#             folder_path = self.path
#             filename_format = self.filename_format
#             channel = None
#
#             if Settings.FILENAME_SUFFIX == "time":
#                 filename_str = filename_format.format(time_str)
#                 path = os.path.join(folder_path, filename_str)
#             else:
#                 filename_str = filename_format.format(self._file_idx[channel])
#                 path = os.path.join(folder_path, filename_str)
#                 while os.path.exists(path):
#                     self._file_idx[channel] += 1
#                     filename_str = filename_format.format(self._file_idx[channel])
#                     path = os.path.join(folder_path, filename_str)
#
#             self.SAVE_EVENT.emit(path)
#             scipy.io.wavfile.write(path, self.sampling_rate, data.astype(Settings.DTYPE))
#         else:
#             raise Exception("When SAVE_CHANNELS_SEPARATELY is on, need channel_folders and filename_format to be lists")
