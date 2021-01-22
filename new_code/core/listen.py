import asyncio
import logging
from functools import partial
import queue
import numpy as np
import sounddevice as sd
import time

# from configure_logging import handler
from core.events import AsyncSignal, Signal
from core.modes import CONTINUOUS, TRIGGERED
from core.ringbuffer import RingBuffer
from core.utils import db_scale

import threading


DTYPE = np.int16
CHUNK = 1024
DETECTION_BUFFER = 0.3

#
# logger = logging.getLogger(__name__)
# logger.addHandler(handler)

def list_devices():
    return sd.query_devices()


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
        self.mic_queue = queue.Queue()
        self._stop_mic_event = None

        self.setup_signals()

    def setup_signals(self):
        self.REC = AsyncSignal()
        self.SETUP = Signal()

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
        if self._stop_mic_event:
            self._stop_mic_event.set()
        if self._stream:
            self._stream.close()

    def _callback(self, in_data, frame_count, time_info, status):
        """Pass on data received on the stream to the REC signal
        """
        self.mic_queue.put(in_data.astype(DTYPE))
        return

    def reset_stream(self):
        if self._stream:
            self._stream.close()

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

        self._stream = self.get_stream(device_info, retry=10)
        self._stream.start()

    def get_stream(self, device_info, timeout=None, retry=5):
        rate = int(device_info["default_samplerate"])

        for _ in range(retry):
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
                time.sleep(0.1)
            else:
                return self._stream

    def run(self):
        """Start an audio input stream and emit metadata

        By using a queue, we decouple the emission of events with the capturing
        of audio data.
        """
        self._stop_mic_event = threading.Event()
        loop = asyncio.get_event_loop()
        # The mic thread is really the main listening thread of the app
        # and will process all downstream events that are emitted.

        # We need to be careful though to not do gui updates on this thread.
        self.mic_thread = threading.Thread(
            target=self._run,
            args=(loop,),
        )
        self.mic_thread.start()

    def _run(self, loop):
        self.reset_stream()
        running = True
        while not self._stop_mic_event.is_set():
            try:
                chunk = self.mic_queue.get(timeout=0.4)
            except:
                running = False
            else:
                asyncio.run_coroutine_threadsafe(self.REC.emit(chunk), loop)
                # loop.call_soon_threadsafe(asyncio.async, self.REC.emit(chunk))


class SoundDetector(object):
    """Emits DETECTED signal when sound crosses a given amplitude threshold

    Emits the channel indexes that the sound was detected on
    """
    DEFAULT_DETECTION_WINDOW = 0.1
    DEFAULT_THRESHOLD = 500.0
    DEFAULT_CROSSINGS_THRESHOLD = 20

    def __init__(
            self,
            detection_window: float=DEFAULT_DETECTION_WINDOW,
            default_threshold: float=DEFAULT_THRESHOLD,
            crossings_threshold: int=DEFAULT_CROSSINGS_THRESHOLD,
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

        self._sampling_rate = None
        self.reset()

        self.DETECTED = Signal()

    def to_config(self):
        return {
            "detection_window": self.detection_window,
            "threshold": self.default_threshold,
            "crossings_threshold": self.crossings_threshold,
        }

    def apply_config(self, config):
        self.detection_window = config.get("detection_window", self.detection_window)
        self.default_threshold = config.get("threshold", self.default_threshold)
        self.crossings_threshold = config.get("crossings_threshold", self.crossings_threshold)
        self.threshold = {}
        if self._sampling_rate is not None:
            self.set_sampling_rate(self._sampling_rate)

    def clear(self):
        self._buffer.clear()

    def reset(self):
        self.clear()
        # Thresholds will be set to default the first time they are accessed
        self.thresholds = {}

    def set_sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate
        self._buffer = RingBuffer(maxlen=int(self.detection_window * sampling_rate))

    def set_channel_threshold(self, channel, threshold):
        self.thresholds[channel] = threshold

    def read_channel_threshold(self, channel):
        if channel not in self.thresholds:
            self.thresholds[channel] = self.default_threshold
        return self.thresholds[channel]

    def receive_data(self, data):
        if self._buffer.maxlen == 0:
            return

        self._buffer.extend(data)
        dat = np.array(self._buffer)

        if not len(dat):
            return

        for ch_idx in range(dat.shape[1]):
            # Thresholds get set here if they are missing
            threshold_crossings = np.nonzero(
                np.diff(np.abs(dat[:, ch_idx]) > self.read_channel_threshold(ch_idx))
            )[0]

            ratio = int(threshold_crossings.size) / self.crossings_threshold
            if ratio > 1:
                self.DETECTED.emit(ch_idx)


class BaseSoundCollector(object):
    """Collect data via trigger and emit signal when data is ready to be saved

    BaseSoundCollector.apply_config(config)
        Set up the sound collector with the configuration parameters. Uses
        the keys "min_file_duration", "max_file_duration", and "buffer_duration"
    BaseSoundCollector.set_sampling_rate(sampling_rate)
        Update the buffer size for the new sampling rate
    BaseSoundCollector.trigger(*args, **kwargs)
        Event handler for triggering recording - implement in subclasses
    BaseSoundCollector.start_recording()
        Set the recording flag to True. This will accumulate data in the ring buffer
    BaseSoundCollector.stop_recording()
        Set the recording flag to False. This may trigger a save on the next
        receive_data call if there is data in the buffer
    BaseSoundCollector.receive_data(chunk: np.ndarray)
        Process a chunk of data by adding it to the current buffer, and emit
        save data via SAVE_READY if necessary
    """
    DEFAULT_MIN_FILE_DURATION = 0.5
    DEFAULT_MAX_FILE_DURATION = 10.0
    DEFAULT_BUFFER_DURATION = 0.2

    def __init__(
            self,
            sampling_rate=None,
            min_file_duration=DEFAULT_MIN_FILE_DURATION,
            max_file_duration=DEFAULT_MAX_FILE_DURATION,
            buffer_duration=DEFAULT_BUFFER_DURATION,
            ):
        """Initialize a SoundCollector instance and buffers

        Params
        ======
        sampling_rate : int
            Sampling rate - if None SoundCollector instance will be inactive.
            Update with SoundCollector.set_sampling_rate()
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

        self._save_buffer = RingBuffer()
        self._trigger_timer = None

        self.min_file_duration = min_file_duration
        self.max_file_duration = max_file_duration
        self.buffer_duration = buffer_duration
        self.set_sampling_rate(sampling_rate)

        self.SAVE_READY = Signal()
        self.RECORDING = Signal()

    def to_config(self):
        return {
            "min_file_duration": self.min_file_duration,
            "max_file_duration": self.max_file_duration,
            "buffer_duration": self.buffer_duration,
        }

    def apply_config(self, config):
        self.min_file_duration = config.get("min_file_duration", self.min_file_duration)
        self.max_file_duration = config.get("max_file_duration", self.max_file_duration)
        self.buffer_duration = config.get("buffer_duration", self.buffer_duration)
        self.set_sampling_rate(self.sampling_rate)

    def set_sampling_rate(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.reset()

    def clear(self):
        self._save_buffer.clear()

    def reset(self):
        if self.sampling_rate is None:
            self.min_file_length = None
            self.max_file_length = None
            self.status["ready"] = False
        else:
            self.min_file_length = int(self.min_file_duration * self.sampling_rate)
            self.max_file_length = int(self.max_file_duration * self.sampling_rate)
            self.status["ready"] = True
            self._save_buffer = RingBuffer(maxlen=int(self.max_file_length) + CHUNK)

    def trigger(self, *args, **kwargs):
        """Initiates the saving of a file

        Silently consumes any arguments passed to it from the caller
        """
        pass

    def start_recording(self):
        if self.status["recording"] is False:
            self.RECORDING.emit(True)
        self.status["recording"] = True

    def stop_recording(self):
        if self.status["recording"] is True:
            self.RECORDING.emit(False)
        self.status["recording"] = False

    def receive_data(self, chunk):
        if not self.status["ready"]:
            return

        self._save_buffer.extend(chunk)

        if self.status["recording"]:
            if len(self._save_buffer) > self.max_file_length:
                data = np.array(self._save_buffer)
                self._save_buffer.clear()
                self._save_buffer.extend(data[self.max_file_length:])
                self.SAVE_READY.emit(data[:self.max_file_length], self.sampling_rate)
        else:
            if self.min_file_length and len(self._save_buffer) > self.min_file_length:
                data = np.array(self._save_buffer)
                self.SAVE_READY.emit(data, self.sampling_rate)
            self._save_buffer.clear()


class TriggeredSoundCollector(BaseSoundCollector):

    def trigger(self, *args, **kwargs):
        """Initiate the recording of data

        Silently consumes any arguments passed to it from the caller
        """
        loop = asyncio.get_running_loop()

        if self._trigger_timer:
            self._trigger_timer.cancel()
            self._trigger_timer = None

        self._trigger_timer = loop.call_later(
            self.buffer_duration,
            self.stop_recording,
        )
        self.start_recording()

    def stop_recording(self):
        super().stop_recording()
        self._trigger_timer = None


class ContinuousSoundCollector(BaseSoundCollector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_recording()


class ToggledSoundCollector(BaseSoundCollector):
    """Triggered sound collector that can be toggled into continuous mode"""

    def __init__(self, is_triggered, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = TRIGGERED if is_triggered else CONTINUOUS
        if self.mode == CONTINUOUS:
            self.start_recording()

    def to_config(self):
        config = super().to_config()
        config["triggered"] = self.mode == TRIGGERED

        return config

    def toggle(self, new_mode):
        self.stop_recording()

        if self.mode == new_mode:
            return
        else:
            self.clear()
            self.mode = new_mode

        if self.mode == CONTINUOUS:
            self.start_recording()

    def trigger(self, *args, **kwargs):
        """Initiate the recording of data

        Silently consumes any arguments passed to it from the caller
        """
        if self.mode == TRIGGERED:
            loop = asyncio.get_running_loop()

            if self._trigger_timer:
                self._trigger_timer.cancel()
                self._trigger_timer = None

            self._trigger_timer = loop.call_later(
                self.buffer_duration,
                self.stop_recording,
            )
            self.start_recording()

    def stop_recording(self):
        super().stop_recording()

        if self._trigger_timer:
            self._trigger_timer.cancel()
            self._trigger_timer = None
