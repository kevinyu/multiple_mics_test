"""Controllers for working with audio streams

save_wav_file()
    - Save data to a directory and filename
class Pathfinder(object)
    - Utility for generating save paths based on current time
class SingleStreamController
class MultiStreamController
class IndependentStreamController(MultiStreamController)
class SynchronizedStreamController(MultiStreamController)
"""

import datetime
import os
from typing import List

import numpy as np
import scipy.io.wavfile

from core.events import Signal
from core.listen import (
    SoundDetector,
    ToggledSoundCollector,
)
from core.modes import (
    CONTINUOUS,
    TRIGGERED
)
from core.utils import db_scale


def save_wav_file(path: str, data, sampling_rate: int):
    """Create path to and save wav file
    """
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(path):
        raise IOError("File {} already exists in {}".format(os.path.basename(path), directory))

    print("scipy.io.wavfile.write({}, {}, {})".format(
        path,
        data.shape,
        sampling_rate,
    ))
    scipy.io.wavfile.write(path, sampling_rate, data)


class Pathfinder(object):
    """Utiltiy for generating save paths and needed directories

    Methods
    =======
    apply_config(config)
    get_dir_and_filename(name, timestamp)
        Return the save directory and save base filename for the given
        name and timestamp, filling in any codes that match Pathfinder.CODES
    """
    CODES = [
        "{name}",
        "{date}",
        "{hour}",
        "{minute}",
        "{second}",
        "{microsecond}",
        "{timestamp}",
        "{year}",
        "{month}",
        "{day}",
    ]

    def to_config(self):
        return {
            "base_dir": self.base_dir,
            "subdirectories": self.subdirectories,
            "filename_format": self.filename_format,
        }

    def apply_config(self, config: dict):
        """Set attributes from config dictionary

        Applies a configuration that includes the keys "base_dir", "subdirectories",
        and "filename_format"
        """
        self.base_dir = config.get("base_dir")
        if not self.base_dir or not os.path.exists(self.base_dir):
            raise ValueError("Base directory {} does not exist".format(self.base_dir))

        self.subdirectories = config.get("subdirectories", [])
        for subdirectory in self.subdirectories:
            if "{" in subdirectory and subdirectory not in self.CODES:
                raise ValueError("Save subdirectory code must be in {}".format(self.CODES))

        self.filename_format = config.get("filename_format", "{name}_{timestamp}")

    def _format_codes(self, name="", timestamp: datetime.datetime=None):
        """Generate a dictionary for string formatting using object and timestamp info

        Each code in Pathfinder.CODES should be defined.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()

        format_dict = {
            "name": name,
            "date": timestamp.strftime("%y%m%d"),
            "year": timestamp.year,
            "month": timestamp.month,
            "day": timestamp.day,
            "hour": timestamp.hour,
            "minute": timestamp.minute,
            "second": timestamp.second,
            "microsecond": timestamp.microsecond,
            "timestamp": self._datetime2str(timestamp),
        }
        return format_dict

    def _datetime2str(self, dt: datetime.datetime) -> str:
        """Convert datetime object to a timestamp integer string

        This timestamp is a number of microseconds from the start of the day
        """
        s = (
            dt.hour * (1000 * 60 * 60) +
            dt.minute * (1000 * 60) +
            dt.second * 1000 +
            dt.microsecond / 1000
        )
        return str(int(s))

    def get_save_path(self, name: str, timestamp: datetime.datetime):
        """Return a full path for the given name and timestamp
        """
        path = self.base_dir
        format_codes = self._format_codes(name=name, timestamp=timestamp)
        for subdirectory_str in self.subdirectories:
            path = os.path.join(path, subdirectory_str.format(**format_codes))

        filename = self.filename_format.format(**format_codes)

        return os.path.join(path, filename)


class SingleStreamController(object):
    """Manages a single stream with a gain, detection, and saving settings

    Links up a microphone to a gain filter.

    Signals
    =======
    OUTPUT (id: int, array)
        Emits the SingleStreamController's id_ as well as the received data
        when data is received on the microphone. The received data has the
        stream's gain applied before being emitted.
    DETECTED (id: int)
        Emits the SingleStreamController's id_ when the detector has been triggered.
    SAVE (path: str, array, sampling_rate: int)
        Emits a request to save audio data at a given path with the given
        sampling_rate

    Methods
    =======
    stop()
        Disconnect events - prepare for deletion
    set_recording_mode(mode)
        Switch between modes.TRIGGERED or modes.CONTINUOUS recording mode
    apply_config(config)

    Attributes
    ==========
    mic: core.listen.Microphone
    saver
    collector
    detector
    name
    gain
    channel
    id_
    """
    def __init__(self, mic, id_: int=0):
        self.id_ = id_

        self.mic = mic
        # self.saver = None
        self.collector = None
        self.detector = None
        self.monitor = True

        self.name = ""
        self.gain = 0
        self.channel = None
        self._is_configured = False

        # Setup signals
        self.mic.REC.connect(self._filter)
        self._FILTERED = Signal()
        self.OUTPUT = Signal()
        self.DETECTED = Signal()
        self.SAVE = Signal()

    def stop(self):
        """Disconnect events - prepare for deletion
        """
        self.mic.REC.disconnect(self._filter)
        self.mic.SETUP.disconnect(self.on_mic_setup)

    def _filter(self, data):
        """Select one channel from data and apply gain

        Emits data to the internal _FILTERED Signal
        """
        # The db_scale function enforces dtype consistency but it can't hurt to
        # double check...
        original_dtype = data.dtype
        out_data = db_scale(data[:, self.channel], self.gain)[:, None]
        self._FILTERED.emit(out_data.astype(original_dtype))

    def set_recording_mode(self, mode: str):
        """Set recording mode between modes.CONTINUOUS and modes.TRIGGERED
        """
        if mode not in (CONTINUOUS, TRIGGERED):
            pass
        else:
            self.collector.toggle(mode)

    def set_monitor(self, monitor: bool):
        self.monitor = monitor

    def apply_config(self, config: dict):
        """Initialize streams from config dict

        Applies a configuration that includes the keys "gain", "channel",
        and "save" and "detect" subconfigurations.
        """
        self.gain = config.get("gain", 0)
        self.channel = config.get("channel")
        self.name = config.get("name", "Ch{}".format(self.channel))
        if self.channel is None:
            raise ValueError("Stream must specific a channel index")

        self.collector = ToggledSoundCollector(config.get("collect", {}).get("triggered"))
        self.collector.apply_config(config.get("collect", {}))

        self.saver = Pathfinder()
        self.saver.apply_config(config.get("save", {}))

        self.detector = SoundDetector()
        self.detector.apply_config(config.get("detect", {}))

        # Link up signals
        self.mic.SETUP.connect(self.on_mic_setup)
        self._FILTERED.connect(self.collector.receive_data)
        self._FILTERED.connect(self.detector.receive_data)
        self._FILTERED.connect(self.on_filtered)
        self.detector.DETECTED.connect(self.collector.trigger)
        self.detector.DETECTED.connect(self.on_detected)
        self.collector.SAVE_READY.connect(self.on_save_ready)

        self._is_configured = True

    def on_mic_setup(self, setup_dict: dict):
        self.collector.set_sampling_rate(setup_dict["rate"])
        self.detector.set_sampling_rate(setup_dict["rate"])

    def on_save_ready(self, data, sampling_rate: int):
        if self.monitor:
            return

        file_timestamp = (
            datetime.datetime.now()
            - (datetime.timedelta(seconds=data.shape[0] / sampling_rate))
        )
        path = self.saver.get_save_path(self.name, file_timestamp)
        path = "{}.wav".format(path)
        self.SAVE.emit(path, data, sampling_rate)

    def on_filtered(self, data):
        """Propogate data"""
        self.OUTPUT.emit(self.id_, data)

    def on_detected(self, _):
        """Propogate detection"""
        self.DETECTED.emit(self.id_)


class MultiStreamController(object):
    """API for managing multiple streams together, organized by index
    """

    def __init__(self, mic):
        self.mic = mic

    def stop(self):
        """Clear out signal observers"""
        raise NotImplementedError

    def get_streams(self) -> List[dict]:
        """Return a list of dictionaries with stream info"""
        raise NotImplementedError

    def set_stream_gain(self, stream_index: int, gain: float):
        """Set the gain of a stream by index
        """
        raise NotImplementedError

    def set_stream_name(self, stream_index: int, name: str):
        """Set the name of a stream by index
        """
        raise NotImplementedError

    def add_stream(self, **config):
        """Add a stream at the last index
        """
        raise NotImplementedError

    def remove_stream(self, stream_index: int):
        """Remove a stream by index
        """
        raise NotImplementedError

    def apply_config(self, config: dict):
        """Apply a config dictionary
        """
        raise NotImplementedError


class IndependentStreamController(MultiStreamController):
    """Manage multiple streams that record and save independently

    Delegates most logic to SingleStreamControllers

    Signals
    =======
    OUTPUT (id: int, array)
        Emits the stream index as well as the received data on that stream
        (with relevant gain/filters applied).
    DETECTED (id: int)
        Emits the stream index that triggered detection

    Methods
    =======
    get_streams(self) -> List[dict]
        Return a list of dictionaries with stream info
    stop()
        Clear out signal observers
    set_stream_gain(self, stream_index: int, gain: float)
        Set the gain of a stream by index
    set_stream_name(self, stream_index: int, name: str)
        Set the name of a stream by index
    add_stream(self, **config)
        Add a stream at the last index
    remove_stream(self, stream_index: int)
        Remove a stream by index
    apply_config(self, config: dict):
        Apply a config dictionary

    Attributes
    ==========
    stream_controllers: List[SingleStreamController]
    mic: core.listen.Microphone
    """
    def __init__(self, mic):
        super().__init__(mic)
        self.stream_controllers: List[SingleStreamController] = []
        self.OUTPUT = Signal()  # Emits stream idx and data
        self.DETECTED = Signal()

    def get_streams(self) -> List[dict]:
        """Return a list of dictionaries with stream info"""
        return [
            {
                "idx": stream.id_,
                "channel": stream.channel,
                "gain": stream.gain,
                "threshold": stream.detector.read_channel_threshold(0),
                "name": stream.name,
                "triggered": stream.collector.mode == TRIGGERED,
                "monitor": stream.monitor,
                "save": None, # IDK
            }
            for stream in self.stream_controllers
        ]

    def stop(self):
        """Clear out signal observers"""
        for stream in self.stream_controllers:
            stream.SAVE.disconnect(save_wav_file)
            stream.OUTPUT.disconnect(self.OUTPUT.emit)
            stream.DETECTED.disconnect(self.DETECTED.emit)
            stream.stop()

    def apply_collector_config(self, collect_config: dict):
        for stream_controller in self.stream_controllers:
            stream_controller.collector.apply_config(collect_config)
            stream_controller.collector.toggle(TRIGGERED if collect_config.get("triggered") else CONTINUOUS)

    @property
    def stream_channels(self):
        return [stream.channel for stream in self.stream_controllers]

    def set_monitor(self, monitor: bool):
        for stream_controller in self.stream_controllers:
            stream_controller.set_monitor(monitor)

    def set_stream_monitor(self, stream_index: int, monitor: bool):
        self.stream_controllers[stream_index].set_monitor(monitor)

    def set_recording_mode(self, mode: str):
        """Set recording mode between modes.CONTINUOUS and modes.TRIGGERED
        """
        if mode not in (CONTINUOUS, TRIGGERED):
            pass
        else:
            for stream_controller in self.stream_controllers:
                stream_controller.collector.toggle(mode)

    def set_stream_recording_mode(self, stream_index: int, mode: str):
        self.stream_controllers[stream_index].collector.toggle(mode)

    def set_threshold(self, threshold: float):
        for stream_controller in self.stream_controllers:
            stream_controller.detector.set_channel_threshold(0, threshold)

    def set_stream_threshold(self, stream_index: int, threshold: float):
        """Set the gain of a stream by index
        """
        self.stream_controllers[stream_index].detector.set_channel_threshold(0, threshold)

    def set_stream_gain(self, stream_index: int, gain: float):
        """Set the gain of a stream by index
        """
        self.stream_controllers[stream_index].gain = gain

    def set_stream_name(self, stream_index: int, name: str):
        """Set the name of a stream by index
        """
        self.stream_controllers[stream_index].name = name

    def add_stream(self, **config):
        """Add a stream at the last index
        """
        self.mic.stop()

        stream = SingleStreamController(mic=self.mic, id_=len(self.stream_controllers))
        stream.apply_config(config)
        stream.OUTPUT.connect(self.OUTPUT.emit)
        stream.DETECTED.connect(self.DETECTED.emit)
        self.stream_contollers.append(stream)

        self.mic.set_channels(self.stream_channels)
        self.mic.reset_stream()

    def remove_stream(self, stream_index: int):
        """Remove a stream by index
        """
        self.stream_controllers[stream_index].stop()
        self.stream_controllers[stream_index].SAVE.disconnect(save_wav_file)
        del self.stream_controllers[stream_index]
        self._renumber()

    def get_buffer_duration(self):
        return self.stream_controllers[0].collector.buffer_duration

    def _renumber(self):
        for i, stream in enumerate(self.stream_controllers):
            stream.id_ = i

    def apply_config(self, config: dict):
        """Apply a config dictionary
        """
        self.mic.stop()
        self.stream_controllers = []
        streams = config.get("streams", [])
        for i, stream_config in enumerate(streams):
            stream = SingleStreamController(mic=self.mic, id_=i)
            stream_config["save"] = config.get("save", {})
            stream.apply_config(stream_config)
            stream.SAVE.connect(save_wav_file)
            stream.OUTPUT.connect(self.OUTPUT.emit)
            stream.DETECTED.connect(self.DETECTED.emit)
            self.stream_controllers.append(stream)

        self.mic.set_channels(self.stream_channels)
        self.mic.reset_stream()

    def to_config(self):
        collect_config = {}
        save_config = self.stream_controllers[0].saver.to_config()
        detect_config = {}

        streams_config = []

        for i, stream_controller in enumerate(self.stream_controllers):
            stream_config = {}
            stream_config["channel"] = stream_controller.channel
            stream_config["name"] = stream_controller.name
            stream_config["gain"] = stream_controller.gain

            stream_config["collect"] = stream_controller.collector.to_config()
            # stream_config["save"] = stream_controller.saver.to_config()
            stream_config["detect"] = stream_controller.detector.to_config()
            stream_config["detect"]["threshold"] = stream_controller.detector.read_channel_threshold(0)

            streams_config.append(stream_config)

        return {
            "save": save_config,
            "detect": detect_config,
            "collect": collect_config,
            "streams": streams_config,
        }


class SynchronizedStreamController(MultiStreamController):
    """Manage multiple streams that record and save together

    Signals
    =======
    OUTPUT (id: int, array)
        Emits the stream index as well as the received data on that stream
        (with relevant gain/filters applied). The data is captured together
        but emitted here separately for convenience (for plotting purposes, etc)
    DETECTED (id: int)
        Emits the stream index that triggered detection

    Methods
    =======
    get_streams(self) -> List[dict]
        Return a list of dictionaries with stream info
    stop()
        Clear out signal observers
    set_stream_gain(self, stream_index: int, gain: float)
        Set the gain of a stream by index
    set_stream_name(self, stream_index: int, name: str)
        Set the name of a stream by index
    add_stream(self, **config)
        Add a stream at the last index
    remove_stream(self, stream_index: int)
        Remove a stream by index
    apply_config(self, config: dict):
        Apply a config dictionary

    Attributes
    ==========
    stream_channels: List[int]
        List of integers for the actual device channels corresponding to each stream
    name_per_stream: List[str]
        Name of each stream
    gain_per_stream: array
        Gain of each stream. Is a numpy array type so it can be multipled as
        an array to incoming data in the _filter method.
    mic: core.listen.Microphone
    saver
    collector
    detector
    """
    def __init__(self, mic):
        super().__init__(mic)

        self.stream_channels: List[int]= []  # Channel indexes per stream
        self.name_per_stream: List[str] = []  # Name of each stream
        self.gain_per_stream = np.array([])  # Gain of each stream

        self.monitor = True

        # Setup signals
        self.mic.REC.connect(self._filter)
        self._FILTERED = Signal()  # Launder the REC signal with the stream's gain
        self.OUTPUT = Signal()  # Emits stream idx and data
        self.DETECTED = Signal()

    def get_streams(self) -> List[dict]:
        """Return a list of dictionaries with stream info"""
        return [
            {
                "idx": i,
                "channel": channel,
                "gain": self.gain_per_stream[i],
                "threshold": self.detector.read_channel_threshold(i),
                "name": self.name_per_stream[i],
                "triggered": self.collector.mode == TRIGGERED,
                "monitor": self.monitor,
                "save": None, # IDK
            }
            for i, channel in enumerate(self.stream_channels)
        ]

    @property
    def name(self):
        """Concatenated names of all streams
        """
        name_parts = []
        for i, name in enumerate(self.name_per_stream):
            name_parts.append("{}_{}".format(i, name))
        return "__".join(name_parts)

    def _filter(self, data):
        """Pick out the relevant channels from the incoming data and apply gain
        """
        # The db_scale function enforces dtype consistency but it can't hurt to
        # double check...
        original_dtype = data.dtype
        out_data = db_scale(data[:, self.stream_channels], self.gain_per_stream)
        self._FILTERED.emit(out_data.astype(original_dtype))

    def stop(self):
        """Clear out signal observers"""
        self.mic.REC.disconnect(self._filter)
        self.mic.SETUP.disconnect(self.on_mic_setup)

    def set_monitor(self, monitor: bool):
        self.monitor = monitor

    def set_stream_monitor(self, stream_index: int, mode: str):
        raise NotImplementedError("Synchronized recordings can't set individual streams to monitor")

    def set_recording_mode(self, mode: str):
        """Set recording mode between modes.CONTINUOUS and modes.TRIGGERED
        """
        if mode not in (CONTINUOUS, TRIGGERED):
            pass
        else:
            self.collector.toggle(mode)

    def apply_collector_config(self, collect_config: dict):
        self.collector.apply_config(collect_config)
        self.collector.toggle(TRIGGERED if collect_config.get("triggered") else CONTINUOUS)

    def set_stream_recording_mode(self, stream_index: int, mode: str):
        raise NotImplementedError("Synchronized recordings can't set individual stream triggers")

    def set_threshold(self, threshold: float):
        for i in range(len(self.stream_channels)):
            self.detector.set_threshold(i, threshold)

    def set_stream_threshold(self, stream_index: int, threshold: float):
        """Set the gain of a stream by index
        """
        self.detector.set_channel_threshold(stream_index, threshold)

    def set_gain(self, gain: float):
        for i in range(len(self.stream_channels)):
            self.gain_per_stream[i] = gain

    def set_stream_gain(self, stream_index: int, gain: float):
        """Set the gain of a stream by index
        """
        self.gain_per_stream[stream_index] = gain

    def set_stream_name(self, stream_index: int, name: str):
        """Set the name of a stream by index
        """
        self.name_per_stream[stream_index] = name

    def add_stream(self, **config):
        """Add a stream at the last index
        """
        self.mic.stop()

        if config.get("channel") is None:
            raise ValueError("Stream must specify a channel index")
        self.stream_channels.append(config["channel"])
        self.name_per_stream.append(config.get("name", "Ch{}".format(config["channel"])))
        self.gain_per_stream = np.concatenate([
            self.gain_per_stream, [config.get("gain", 0)]
        ])

        # self.collector.clear()  # These might not be necessary, as the mic.SETUP event may trigger these
        # self.detector.clear()
        self.mic.set_channels(self.stream_channels)
        self.mic.reset_stream()

    def remove_stream(self, stream_index: int):
        """Remove a stream by index
        """
        if stream_index >= len(self.stream_channels):
            raise ValueError("stream_index out of range")

        self.mic.stop()
        del self.stream_channels[stream_index]
        del self.name_per_stream[stream_index]
        np.delete(self.gain_per_stream, stream_index)

        # self.collector.clear()  # These might not be necessary, as the mic.SETUP event may trigger these
        # self.detector.clear()
        self.mic.set_channels(self.stream_channels)
        self.mic.reset_stream()

    def apply_config(self, config: dict):
        """Apply a config dictionary

        Initializes the collector and detector objects and hooks up events
        """
        self.mic.stop()

        streams = config.get("streams", [])
        self.stream_channels = []
        self.name_per_stream = []
        self.gain_per_stream = np.zeros(len(streams))

        for i, stream in enumerate(streams):
            if stream.get("channel") is None:
                raise ValueError("Stream at index {} must specific a channel index".format(i))
            self.stream_channels.append(stream["channel"])
            self.name_per_stream.append(stream.get("name", "Ch{}".format(stream["channel"])))
            self.gain_per_stream[i] = stream.get("gain", 0)

        self.collector = ToggledSoundCollector(config.get("collect", {}).get("triggered"))
        self.collector.apply_config(config.get("collect", {}))

        self.saver = Pathfinder()
        self.saver.apply_config(config.get("save", {}))

        self.detector = SoundDetector()
        self.detector.apply_config(config.get("detect", {}))
        for i, stream in enumerate(streams):
            if stream.get("detect", {}).get("threshold"):
                self.detector.set_channel_threshold(i, stream["detect"]["threshold"])

        # Link up signals
        self.mic.SETUP.connect(self.on_mic_setup)
        self._FILTERED.connect(self.collector.receive_data)
        self._FILTERED.connect(self.detector.receive_data)
        self._FILTERED.connect(self.on_filtered)
        self.detector.DETECTED.connect(self.collector.trigger)
        self.detector.DETECTED.connect(self.DETECTED.emit)
        self.collector.SAVE_READY.connect(self.on_save_ready)

        self.mic.set_channels(self.stream_channels)
        self.mic.reset_stream()

    def get_buffer_duration(self):
        return self.collector.buffer_duration

    def on_mic_setup(self, setup_dict: dict):
        self.collector.set_sampling_rate(setup_dict["rate"])
        self.detector.set_sampling_rate(setup_dict["rate"])

    def on_save_ready(self, data, sampling_rate: int):
        if self.monitor:
            return

        file_timestamp = (
            datetime.datetime.now()
            - (datetime.timedelta(seconds=data.shape[0] / sampling_rate))
        )
        path = self.saver.get_save_path(self.name, file_timestamp)
        path = "{}.wav".format(path)
        save_wav_file(path, data, sampling_rate)

    def on_filtered(self, data):
        for i in range(data.shape[1]):
            self.OUTPUT.emit(i, data[:, i:i+1])

    def to_config(self):
        collect_config = self.collector.to_config()
        save_config = self.saver.to_config()
        detect_config = self.detector.to_config()

        streams_config = []

        for i in range(len(self.stream_channels)):
            stream_config = {}
            stream_config["channel"] = self.stream_channels[i]
            stream_config["name"] = self.name_per_stream[i]
            stream_config["gain"] = self.gain_per_stream[i]
            stream_config["collect"] = {}
            # stream_config["save"] = {}
            stream_config["detect"] = {
                "threshold": self.detector.read_channel_threshold(i)
            }
            streams_config.append(stream_config)

        return {
            "save": save_config,
            "detect": detect_config,
            "collect": collect_config,
            "streams": streams_config,
        }
