import asyncio
import datetime
import logging
import os
from functools import partial

import numpy as np
import scipy.io.wavfile

from configure_logging import handler
from events import Signal
from listen import (
    Microphone,
    SoundDetector,
    GainFilter,
    SoundCollector,
    FilteredMicrophone,
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

    Config Example
    ==============
    gain: 10
    channel: 1
    name: "Subject1"
    save:
        min_file_duration: 1.0
        max_file_duration: 10.0
        buffer_duration: 0.2
        base_dir: "/data"
        subdirectories: ["{name}", "{date}", "{hour}"]
        filename_format: "{name}_{timestamp}"
    detect:
        threshold: 1000.0
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
        self.OUT = Signal()

    def _filter(self, data):
        out_data = db_scale(data[:, self.channel], self.gain)
        self.OUT.emit(data)

    def apply_config(self, config):
        self.gain = config.get("gain", 0)
        self.channel = config.get("channel")
        self.name = config.get("name", "Ch{}".format(self.channel))
        if self.channel is None:
            raise ValueError("Stream must specific a channel index")

        self.collector = SoundCollector()
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

# class SynchronizedStreamManager(object):
#     def __init__(self, mic):
#         self.mic = mic
#         self,detectors_per_channel = {}
#         self.saver = None
#
#         self.gain_per_channel = {}
#         self._is_configured = False
#
#         # Setup signals
#         self.mic.REC.connect(self._filter)
#         self.OUT = Signal()
#
#     def _filter(self, data):
#         for i in range(data.shape[1]):
#             data[:, i] = db_scale(data[:, i], self.gain_per_channel.get(i, 0))
#         self.OUT.emit(data)
#
#     def apply_config(self, config):
#         # Config contains the info on each stream
#         self.saver = SoundSaver()
#         self.saver.apply_config(config.get("saving", {}))
#
#         for stream_config in config["streams"]:
#             channel = stream_config["channel"]
#             if channel in self.gain_per_channel:
#                 raise ValueError("Configuration has repeated channel")
#
#             stream_detector = SoundDetector()
#             stream_detector.apply_config(stream_config.get("detection"), {})
#
#             self.detectors_per_channel[channel] = stream_detector
#             self.gain_per_channel[channel] = stream_config.get("gain", 0)
#             self.channels.append(channel)
#
#         # Hookup events
#         self.mic.SETUP.connect(self.on_mic_setup)
#         self.OUT.connect(self.saver.receive_data)
#         for detector in self.detectors_per_channel.value():
#             self.OUT.connect(detector.receive_data)
#             detector.DETECTED.connect(self.saver.trigger())
#
#         self.channels = sorted(self.channels)
#         self._is_configured = True
#
#     def on_mic_setup(self, setup_dict):
#         self.saver.set_sampling_rate(setup_dict["rate"])
#         for detector in self.detectors_per_channel.value():
#             ddetector.set_sampling_rate(setup_dict["rate"])
#
#     def save_data(self, data):
#         pass



class CoreApp(object):
    """Configurable background recording app
    """
    def __init__(self, config):
        # The mic must be instantiated within the main asyncio loop coroutine!
        pass

    def apply_config(self, config):
        device_index = config.get("device_index")
        streams = config.get("streams", [])

        self.mic = Microphone(device_index=device_index)

        channels = []
        for stream_config in streams:
            channels.append(stream_config["channel"])

        self.pb = DetectionsPowerbar(decay_time=0.2, max_value=1e4, channels=channels)
        for stream_config, channel in zip(streams, channels):
            stream_manager = StreamManager(mic=self.mic)
            stream_manager.apply_config(stream_config)
            stream_manager.detector.DETECTED.connect(lambda ch: self.pb.set_detected(channel))

        self.mic.REC.connect(self.echo)
        self.mic.SETUP.connect(lambda setup_dict: self.pb.set_channels(setup_dict["n_channels"]))
        self.mic.set_channels(np.max(channels) + 1)

    def echo(self, data):
        for i in range(data.shape[1]):
            self.pb.set_channel_value(i, np.max(np.abs(data[:, i])))
        self.pb.print()

    async def run(self, config):
        self.apply_config(config)

        try:
            await self.mic.run()
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self.mic.stop()




config = {
    "device_index": 6,
    "streams": [
        {
            "channel": 0,
            "gain": 20,
            "name": "KevinsTest",
            "save": {
                "min_file_duration": 0.5,
                "max_file_duration": 10.0,
                "buffer_duration": 0.2,
                "base_dir": "temp",
                "subdirectories": ["{name}", "{date}"],
                "filename_format": "{name}_{date}_{timestamp}"
            },
            "detect": {
                "threshold": 1000.0
            }
        }
    ]
}


def example_app(*args):
    app = CoreApp(config)
    asyncio.run(app.run(config))
