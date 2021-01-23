import asyncio
import datetime
import logging
import os
import time
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
    ToggledSoundCollector,
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









#
# class SynchronizedStreamManager(MultiStreamController):
#
#     def __init__(self, mic):
#         super().__init__(mic)
#         self.collector = None
#         self.detector = None
#
#         # These channels are more like "streams" since they can duplicate channels
#         self.stream_channels = []  # Channel indexes per stream
#         self.name_per_stream = []  # Name of each stream
#         self.gain_per_stream = np.array([])  # Gain of each stream
#
#         self._is_configured = False
#
#         # Setup signals
#         self.mic.REC.connect(self._filter)
#         self.OUT = Signal()  # Launder the REC signal with the stream's gain
#
#     @property
#     def name(self):
#         name_parts = []
#         for i, name in enumerate(self.name_per_stream):
#             name_parts.append("{}_{}".format(i, name))
#         return "__".join(name_parts)
#
#     def _filter(self, data):
#         # The db_scale function enforces dtype consistency but it can't hurt to
#         # double check...
#         original_dtype = data.dtype
#         out_data = db_scale(data[:, self.stream_channels], self.gain_per_stream)
#         self.OUT.emit(out_data.astype(original_dtype))
#
#     def apply_config(self, config):
#         self.mic.stop()
#
#         streams = config.get("streams", [])
#         self.stream_channels = []
#         self.name_per_stream = []
#         self.gain_per_stream = np.zeros(len(streams))
#
#         for i, stream in enumerate(streams):
#             if stream.get("channel") is None:
#                 raise ValueError("Stream at index {} must specific a channel index".format(i))
#             self.stream_channels.append(stream["channel"])
#             self.name_per_stream.append(stream.get("name", "Ch{}".format(stream["channel"])))
#             self.gain_per_stream[i] = stream.get("gain", 0)
#
#         self.collector = ToggledSoundCollector(config.get("save", {}).get("triggered"))
#         self.collector.apply_config(config.get("save", {}))
#
#         self.saver = Pathfinder()
#         self.saver.apply_config(config.get("save", {}))
#
#         self.detector = SoundDetector()
#         self.detector.apply_config(config.get("detect", {}))
#
#         # Link up signals
#         self.mic.SETUP.connect(self.on_mic_setup)
#         self.OUT.connect(self.collector.receive_data)
#         self.OUT.connect(self.detector.receive_data)
#         self.detector.DETECTED.connect(self.collector.trigger)
#         self.collector.SAVE_READY.connect(self.save_data)
#
#         self._is_configured = True
#
#         self.mic.set_channels(self.stream_channels)
#         self.mic.reset_stream()


# class CoreApp(object):
#     """Configurable background recording app
#     """
#     def __init__(self, config):
#         # The mic must be instantiated within the main asyncio loop coroutine!
#         self._config = config
#
#     def apply_config(self, config):
#         self._config = config
#
#         device_name = config.get("device_name")
#         streams = config.get("streams", [])
#
#         # Apply defaults
#         for stream in streams:
#             if "gain" not in stream:
#                 stream["gain"] = config.get("gain")
#             for key, val in config.get("save", {}).items():
#                 if key not in stream.get("save", {}):
#                     if "save" not in stream:
#                         stream["save"] = {}
#                     stream["save"][key] = val
#             for key, val in config.get("detect", {}).items():
#                 if key not in stream.get("detect", {}):
#                     if "detect" not in stream:
#                         stream["detect"] = {}
#                     stream["detect"][key] = val
#
#         self.mic = Microphone(device_name=device_name)
#
#         channels = []
#         for stream_config in streams:
#             channels.append(stream_config["channel"])
#
#         self.pb = DetectionsPowerbar(decay_time=0.2, max_value=1e4, channels=len(channels))
#
#         if config["synchronized"] is True:
#             stream_manager = SynchronizedStreamManager(mic=self.mic)
#             stream_manager.apply_config(config)
#             stream_manager.detector.DETECTED.connect(partial(self.pb.set_detected, None))
#             stream_manager.OUT.connect(self.echo)
#         else:
#             for i, stream_config in enumerate(streams):
#                 stream_manager = StreamManager(mic=self.mic)
#                 stream_manager.apply_config(stream_config)
#                 stream_manager.detector.DETECTED.connect(partial(self.pb.set_detected, i))
#                 stream_manager.OUT.connect(partial(self.echo_channel, i))
#
#         self.mic.set_channels(channels)
#
#     def echo(self, data):
#         for i in range(data.shape[1]):
#             self.pb.set_channel_value(i, np.max(np.abs(data[:, i])))
#         self.pb.print()
#
#     def echo_channel(self, i, data):
#         self.pb.set_channel_value(i, np.max(np.abs(data)))
#         self.pb.print()
#
#     async def run(self):
#         self.apply_config(self._config)
#
#         try:
#             await self.mic.run()
#         except KeyboardInterrupt:
#             logger.info("Shutting down")
#         finally:
#             self.mic.stop()


def example_app(config_file):
    with open(config_file, "r") as yaml_config:
        config = yaml.load(yaml_config)
    print(config["devices"])
    app = CoreApp(config["devices"][0])
    asyncio.run(app.run())



from qasync import QEventLoop, asyncSlot, asyncClose
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton,
    QVBoxLayout)
from PyQt5 import QtWidgets as widgets



class CoreApp(object):
    """Configurable abstraction for recording app
    """

    def __init__(self):
        # The mic must be instantiated within the main asyncio loop coroutine!
        # self._config = config
        self.mic = None
        self._config = {}
        self.stream_controller = None

    def apply_config(self, config):
        self.stop()

        self._config = config
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

        config["streams"] = streams

        self.mic = Microphone(device_name=device_name)

        if config["synchronized"] is True:
            self.stream_controller = SynchronizedStreamController(mic=self.mic)
        else:
            self.stream_controller = IndependentStreamController(mic=self.mic)
        self.stream_controller.apply_config(config)

    def toggle_synchronized(self):
        self.stream_controller.stop()
        del self.stream_controller
        self._config["synchronized"] = not self._config["synchronized"]
        self.apply_config(self._config)

    def run(self):
        return self.mic.run()

    def stop(self):
        if self.mic:
            self.mic.stop()


from PyQt5.QtCore import QTimer, pyqtSignal, QObject, QThread, pyqtSlot, Qt


class QApp(QObject, CoreApp):
    RECV = pyqtSignal([int, object])  # Receive data on stream index
    SETUP_STREAMS = pyqtSignal(object)
    DETECTED = pyqtSignal([int])  # Report detection on stream index

    def apply_config(self, config):
        super().apply_config(config)
        self.stream_controller.OUTPUT.connect(self.RECV.emit)
        self.stream_controller.DETECTED.connect(self.DETECTED.emit)
        self.SETUP_STREAMS.emit(config["streams"])


class MainWindow(widgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Recorder"
        self._mic_task = None

        # Container for main application logic
        self.app = QApp()
        self.init_ui()

        self.app.RECV.connect(self.update_label)
        self.app.SETUP_STREAMS.connect(self.on_setup_streams)
        self.app.DETECTED.connect(self.update_detections)

        self.label_texts = []
        self.label_detects = []

    @asyncClose
    async def closeEvent(self, event):
        self.app.stop()
        self._mic_task.cancel()
        return

    def init_ui(self):
        main_frame = widgets.QFrame(self)
        self.layout = QVBoxLayout()
        self.scroll_area = widgets.QScrollArea(self)
        # self.scroll_area.setFixedHeight(650)
        # self.scroll_area.setFixedWidth(800)
        self.scroll_area.setWidgetResizable(True)
        main_frame.setLayout(self.layout)
        # self.scroll_area.setWidget(self.recording_window)
        self.detect_label = QLabel("Detecte", self)
        self.layout.addWidget(self.detect_label)

        self.label = QLabel("Poop", self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.scroll_area)

        self.btn = QPushButton("Push me ", self)
        self.btn.clicked.connect(self.app.stop)

        self.layout.addWidget(self.btn)
        self.setCentralWidget(main_frame)

    def update_label(self, stream_idx, data):
        self.label_texts[stream_idx] = "{} {} {}".format(stream_idx, np.min(data), np.max(data))
        self.update()

    def update(self):
        now = time.time()
        detects = [then and now - then < 0.2 for then in self.label_detects]
        parts = []
        for detect, txt in zip(detects, self.label_texts):
            parts.append("{} {}".format("+" if detect else "-", txt))

        self.label.setText("\n".join(parts))

    def update_detections(self, stream_idx):
        self.label_detects[stream_idx] = time.time()
        self.update()

    def on_setup_streams(self, stream_configs):
        self.label_texts = [""] * len(stream_configs)
        self.label_detects = [None] * len(stream_configs)

    def set_config(self, config):
        self.app.apply_config(config)
        self._mic_task = asyncio.create_task(self.app.mic.run())
        # self.run_mic()


def run_pyqt_app(config_file):
    import sys

    with open(config_file, "r") as yaml_config:
        config = yaml.load(yaml_config)
    print(config["devices"])

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    mainWindow = MainWindow()
    mainWindow.set_config(config["devices"][0])
    mainWindow.show()

    with loop:
        loop.run_forever()
