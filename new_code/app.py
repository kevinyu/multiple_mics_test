import asyncio
import datetime
import logging
import os
from functools import partial

import numpy as np
import scipy.io.wavfile
import yaml

from configure_logging import handler
from events import Signal
from listen import (
    Microphone,
    SoundDetector,
    ContinuousSoundCollector,
    TriggeredSoundCollector,
)
from visualize import Powerbar, DetectionsPowerbar
from utils import db_scale


logger = logging.getLogger(__name__)
logger.addHandler(handler)


async def main_basic(device_index):
    pb = Powerbar(max_value=5000, channels=2)

    def echo(data):
        for i in range(data.shape[1]):
            pb.set_channel_value(i, np.max(np.abs(data[:, i])))
        pb.print()

    mic = Microphone(device_index=device_index)
    mic.SETUP.connect(lambda s: pb.set_channels(s["n_channels"]))
    mic.REC.connect(echo)
    mic.set_channels(2)

    await mic.run()

    try:
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        mic.stop()


async def main_detection(device_index, channels=1):
    """Run a detector for signal on given channel
    """
    pb = DetectionsPowerbar(decay_time=0.2, max_value=1e4, channels=channels)
    mic = Microphone(device_index=device_index)
    gain_filter = GainFilter({i: 20 for i in range(channels)})
    detector = SoundDetector(
        detection_window=0.3,
        default_threshold=1000,
        crossings_threshold=20,
    )

    def echo(data):
        for i in range(data.shape[1]):
            pb.set_channel_value(i, np.max(np.abs(data[:, i])))
        pb.print()

    def configure(setup_dict):
        detector.reset()
        pb.set_channels(setup_dict["n_channels"])
        detector.set_sampling_rate(setup_dict["rate"])

    filtered_mic = mic.apply(gain_filter)

    mic.SETUP.connect(configure)
    filtered_mic.REC.connect(echo)
    filtered_mic.REC.connect(detector.receive_data)
    detector.DETECTED.connect(pb.set_detected)

    mic.set_channels(channels)

    try:
        await mic.run()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        mic.stop()


async def main_savefn(device_index, channels=1):
    """Run a detector for signal on given channel
    """
    pb = DetectionsPowerbar(decay_time=0.2, max_value=1e4, channels=channels)
    saver = SoundCollector()
    mic = Microphone(device_index=device_index)
    detector = SoundDetector(
        detection_window=0.3,
        default_threshold=1000,
        crossings_threshold=20,
    )

    def echo(data):
        for i in range(data.shape[1]):
            pb.set_channel_value(i, np.max(np.abs(data[:, i])))
        pb.print()

    def configure(setup_dict):
        detector.reset()
        pb.set_channels(setup_dict["n_channels"])
        detector.set_sampling_rate(setup_dict["rate"])
        saver.set_sampling_rate(setup_dict["rate"])

    gain_filter = GainFilter({i: 20 for i in range(channels)})
    filtered_mic = mic.apply(gain_filter)

    mic.SETUP.connect(configure)
    filtered_mic.REC.connect(echo)
    filtered_mic.REC.connect(detector.receive_data)
    filtered_mic.REC.connect(saver.receive_data)
    detector.DETECTED.connect(pb.set_detected)
    detector.DETECTED.connect(lambda x: saver.trigger())
    saver.SAVE_READY.connect(print)

    mic.set_channels(channels)

    try:
        await mic.run()
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        mic.stop()


def example_app_1(device_index, channels):
    asyncio.run(main_savefn(device_index, channels))


class Pathfinder(object):
    subdirectory_codes = [
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

    def apply_config(self, config):
        self.base_dir = config.get("base_dir")
        if not self.base_dir or not os.path.exists(self.base_dir):
            raise ValueError("Base directory {} does not exist".format(self.base_dir))

        self.subdirectories = config.get("subdirectories", [])
        for subdirectory in self.subdirectories:
            if "{" in subdirectory and subdirectory not in self.subdirectory_codes:
                raise ValueError("Save subdirectory code must be in {}".format(self.subdirectory_codes))

        self.filename_format = config.get("filename_format", "{name}_{timestamp}")

    def format_codes(self, name="", timestamp=None):
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

    def _datetime2str(self, dt):
        s = (
            dt.hour * (1000 * 60 * 60) +
            dt.minute * (1000 * 60) +
            dt.second * 1000 +
            dt.microsecond / 1000
        )
        return str(int(s))

    def get_dir_and_filename(self, name, timestamp):
        path = self.base_dir
        format_codes = self.format_codes(name=name, timestamp=timestamp)
        for subdirectory_str in self.subdirectories:
            path = os.path.join(path, subdirectory_str.format(**format_codes))

        filename = self.filename_format.format(**format_codes)

        return path, filename


class StreamManager(object):
    """Manages a stream with a gain, detection, and saving settings

    Links up a microphone to a gain filter, and saves file to a stream.
    """
    def __init__(self, mic):
        self.mic = mic
        self.saver = None
        self.collector = None
        self.detector = None

        self.name = ""
        self.gain = 0
        self.channel = None
        self._is_configured = False

        # Setup signals
        self.mic.REC.connect(self._filter)
        self.OUT = Signal()  # Launder the REC signal with the stream's gain

    def stop(self):
        self.mic.REC.disconnect(self._filter)
        self.mic.SETUP.disconnect(self.on_mic_setup)

    def _filter(self, data):
        # The db_scale function enforces dtype consistency but it can't hurt to
        # double check...
        original_dtype = data.dtype
        out_data = db_scale(data[:, self.channel], self.gain)[:, None]
        self.OUT.emit(out_data.astype(original_dtype))

    def apply_config(self, config):
        self.gain = config.get("gain", 0)
        self.channel = config.get("channel")
        self.name = config.get("name", "Ch{}".format(self.channel))
        if self.channel is None:
            raise ValueError("Stream must specific a channel index")

        if config.get("save", {}).get("triggered"):
            self.collector = TriggeredSoundCollector()
        else:
            self.collector = ContinuousSoundCollector()
        self.collector.apply_config(config.get("save", {}))

        self.saver = Pathfinder()
        self.saver.apply_config(config.get("save", {}))

        self.detector = SoundDetector()
        self.detector.apply_config(config.get("detect", {}))

        # Link up signals
        self.mic.SETUP.connect(self.on_mic_setup)
        self.OUT.connect(self.collector.receive_data)
        self.OUT.connect(self.detector.receive_data)
        self.detector.DETECTED.connect(self.collector.trigger)
        self.collector.SAVE_READY.connect(self.save_data)

        self._is_configured = True

    def on_mic_setup(self, setup_dict):
        self.collector.set_sampling_rate(setup_dict["rate"])
        self.detector.set_sampling_rate(setup_dict["rate"])

    def save_data(self, data, sampling_rate):
        file_timestamp = (
            datetime.datetime.now()
            - (datetime.timedelta(seconds=data.shape[0] / sampling_rate))
        )
        save_dir, filename = self.saver.get_dir_and_filename(self.name, file_timestamp)

        filename = "{}.wav".format(filename)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        full_path = os.path.join(save_dir, filename)
        if os.path.exists(full_path):
            raise IOError("File {} already exists in {}".format(filename, save_dir))

        print("scipy.io.wavfile.write({}, {}, {})".format(
            full_path,
            data.shape,
            sampling_rate,
        ))
        scipy.io.wavfile.write(full_path, sampling_rate, data)


class SynchronizedStreamManager(StreamManager):

    def __init__(self, mic):
        self.mic = mic
        self.saver = None
        self.collector = None
        self.detector = None

        # These channels are more like "streams" since they can duplicate channels
        self.stream_channels = []  # Channel indexes per stream
        self.name_per_stream = []  # Name of each stream
        self.gain_per_stream = np.array([])  # Gain of each stream

        self._is_configured = False

        # Setup signals
        self.mic.REC.connect(self._filter)
        self.OUT = Signal()  # Launder the REC signal with the stream's gain

    @property
    def name(self):
        name_parts = []
        for i, name in enumerate(self.name_per_stream):
            name_parts.append("{}_{}".format(i, name))
        return "__".join(name_parts)

    def _filter(self, data):
        # The db_scale function enforces dtype consistency but it can't hurt to
        # double check...
        original_dtype = data.dtype
        out_data = db_scale(data[:, self.stream_channels], self.gain_per_stream)
        self.OUT.emit(out_data.astype(original_dtype))

    def apply_config(self, config):
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

        if config.get("save", {}).get("triggered"):
            self.collector = TriggeredSoundCollector()
        else:
            self.collector = ContinuousSoundCollector()
        self.collector.apply_config(config.get("save", {}))

        self.saver = Pathfinder()
        self.saver.apply_config(config.get("save", {}))

        self.detector = SoundDetector()
        self.detector.apply_config(config.get("detect", {}))

        # Link up signals
        self.mic.SETUP.connect(self.on_mic_setup)
        self.OUT.connect(self.collector.receive_data)
        self.OUT.connect(self.detector.receive_data)
        self.detector.DETECTED.connect(self.collector.trigger)
        self.collector.SAVE_READY.connect(self.save_data)

        self._is_configured = True


class CoreApp(object):
    """Configurable background recording app
    """
    def __init__(self, config):
        # The mic must be instantiated within the main asyncio loop coroutine!
        self._config = config

    def apply_config(self, config):
        device_name = config.get("device_name")
        streams = config.get("streams", [])

        # Apply defaults
        for stream in streams:
            if "gain" not in stream:
                stream["gain"] = config.get("gain")
            for key, val in config.get("save", {}).items():
                if key not in stream.get("save", {}):
                    if "save" not in stream:
                        stream["save"] = {}
                    stream["save"][key] = val
            for key, val in config.get("detect", {}).items():
                if key not in stream.get("detect", {}):
                    if "detect" not in stream:
                        stream["detect"] = {}
                    stream["detect"][key] = val

        self.mic = Microphone(device_name=device_name)

        channels = []
        for stream_config in streams:
            channels.append(stream_config["channel"])

        self.pb = DetectionsPowerbar(decay_time=0.2, max_value=1e4, channels=len(channels))

        if config["synchronized"] is True:
            stream_manager = SynchronizedStreamManager(mic=self.mic)
            stream_manager.apply_config(config)
            stream_manager.detector.DETECTED.connect(partial(self.pb.set_detected, None))
            stream_manager.OUT.connect(self.echo)
        else:
            for i, stream_config in enumerate(streams):
                stream_manager = StreamManager(mic=self.mic)
                stream_manager.apply_config(stream_config)
                stream_manager.detector.DETECTED.connect(partial(self.pb.set_detected, i))
                stream_manager.OUT.connect(partial(self.echo_channel, i))

        self.mic.set_channels(channels)

    def echo(self, data):
        for i in range(data.shape[1]):
            self.pb.set_channel_value(i, np.max(np.abs(data[:, i])))
        self.pb.print()

    def echo_channel(self, i, data):
        self.pb.set_channel_value(i, np.max(np.abs(data)))
        self.pb.print()

    async def run(self):
        self.apply_config(self._config)

        try:
            await self.mic.run()
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.mic.stop()


def example_app(config_file):
    with open(config_file, "r") as yaml_config:
        config = yaml.load(yaml_config)
    print(config["devices"])
    app = CoreApp(config["devices"][0])
    asyncio.run(app.run())
