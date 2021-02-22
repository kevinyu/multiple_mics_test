import asyncio
import datetime
import logging
import os
import time
from functools import partial

import numpy as np
import scipy.io.wavfile
import yaml

from core.app import AppController
from core.events import Signal
from core.listen import (
    Microphone,
    SoundDetector,
    ContinuousSoundCollector,
    TriggeredSoundCollector,
    ToggledSoundCollector,
)
from core.utils import db_scale
from visualize import Powerbar, DetectionsPowerbar


logger = logging.getLogger(__name__)
logging.basicConfig()


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


def headless_app(config_file, save_on=False):
    app = AppController()
    with open(config_file, "r") as yaml_config:
        config = yaml.load(yaml_config)

    print("Running app with config:\n{}".format(config))
    app.apply_config(config)
    app.set_monitor(not save_on)
    app.run()

    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("Closing program")
        app.stop()
