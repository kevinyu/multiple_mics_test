"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""
from __future__ import print_function

import matplotlib
matplotlib.use("agg")
import os
import sys
import time
from threading import Thread

import numpy as np
import pyaudio
import pygame as pg
import scipy.io.wavfile

from soundsig.signal import bandpass_filter


CHUNK = 1024 * 4
WIDTH = 2
GUI_WIDTH = 640
GUI_HEIGHT = 480

RATE = 30000
SILENT_BUFFER = 1.0
CROSSINGS_THRESHOLD = 0.2 * 200 # (heuristic: 200 crossings per second during sound)
WHITE = (255,255,255) #RGB
BLACK = (0,0,0) #RGB


class MessageType(object):
    RECORDING = "recording"
    SOUND_RECEIVED = "sound_received"
    RECORDING_CAPTURED = "recording_captured"
    ABOVE_THRESHOLD = "above_threshold"
    CLOSING = "closing"
    LISTENING = "listening"
    LOOP = "loop"


class PubSub(object):
    """Simple message handler to coordinate events across services

    Might be a reimplementation of something that already exists....
    but whatever.
    """

    def __init__(self):
        self.handlers = {}

    def emit(self, message_type, **kwargs):
        for handler in self.handlers.get(message_type, []):
            handler.handle(message_type, **kwargs)

    def subscribe(self, message_type, sub):
        if message_type not in self.handlers:
            self.handlers[message_type] = [sub]
        else:
            self.handlers[message_type].append(sub)

    def unsubscribe(self, message_type, sub):
        self.handlers[message_type].remove(sub)


class Timer(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.time_start = time.time()

    def time(self):
        return time.time() - self.time_start


class AppService(object):
    """Abstract class for services of pygame app"""
    def __init__(self, app):
        self.app = app

    def emit(self, message_type, **kwargs):
        self.app.pubsub.emit(message_type, **kwargs)


class SoundDetector(AppService):

    def __init__(
                self,
                app,
                buffer_duration=SILENT_BUFFER,
                max_time=10.0,
                detection_window=1024 * 4,
                amp_threshold=500,
            ):
        super(SoundDetector, self).__init__(app)
        self.recording = False
        self.buffer_duration = buffer_duration
        self.amp_threshold = amp_threshold
        self.detection_window = detection_window
        self.max_time = max_time
        self.recording_timer = Timer()
        self.threshold_timer = Timer()

    def handle(self, message_type, **kwargs):
        if message_type is MessageType.SOUND_RECEIVED:
            buffer = kwargs["data"]
            self.process(buffer)

    def process(self, buffer):
        """Check for threhsold crossing in recently collected data"""
        if self.recording is False and self.above_threshold(buffer):
            self.recording = True
            self.recording_timer.reset()
            self.threshold_timer.reset()
            self.emit(MessageType.RECORDING, value=True)
        elif self.recording is True and self.above_threshold(buffer):
            self.threshold_timer.reset()
        elif self.recording is True and (
                    self.threshold_timer.time() > self.buffer_duration or
                    self.recording_timer.time() > self.max_time
                ):
            self.recording = False
            self.emit(MessageType.RECORDING, value=False)

    def above_threshold(self, buffer):
        detection_window = buffer[-self.detection_window:]
        data = bandpass_filter(detection_window.T, RATE, 100, 10000).T

        ratios = {}
        did_exceed_threshold = False
        for ch_idx in range(self.app.channels):
            threshold_crossings = np.nonzero(
                np.diff(detection_window[:, ch_idx] > self.amp_threshold)
            )[0]
            ratios[ch_idx] = int(threshold_crossings.size) / CROSSINGS_THRESHOLD

            if ratios[ch_idx] > 1:
                did_exceed_threshold = True

        self.emit(
            MessageType.ABOVE_THRESHOLD,
            ratios=ratios
        )

        return did_exceed_threshold


class MicPanel(object):
    def __init__(self, x, y, width=100, height=100):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_radius = 20
        self.max_radius = 40
        self.scale = 100.0
        self.surface = pg.Surface((self.width, self.height))

    def draw(self, surface):
        return surface.blit(self.surface, (self.x, self.y))

    def set_data(self, data, offset):
        self.surface.fill(WHITE)
        x = np.linspace(self.max_radius * 2, self.width, len(data))
        y = self.height // 2 - (data / self.scale)
        y[:-offset] = y[offset:]
        y[-offset:] = 0
        # line = pg.draw.lines(
        #     self.surface,
        #     BLACK,
        #     False,
        #     list(zip(*(x[::1000], y[::1000])))
        # )

    def set_mic_level(self, scale):
        radius = self.min_radius + int(scale * (self.max_radius - self.min_radius))

        pg.draw.circle(
            self.surface,
            (200, 100, 100),
            (self.max_radius, self.max_radius),
            radius
        )
        pg.draw.circle(
            self.surface,
            BLACK,
            (self.max_radius, self.max_radius),
            self.min_radius
        )

class GUI(AppService):

    def __init__(self, app):
        super(GUI, self).__init__(app)
        self.screen = pg.display.set_mode((GUI_WIDTH, GUI_HEIGHT), 0, 32)
        pg.display.set_caption("Live Mic Recording")
        self.screen.fill(WHITE)
        self.screen.set_alpha(None)

        self.mic_panels = [
            MicPanel(0, 120 * i, 600, 100)
            for i in range(app.channels)
        ]
        self.last_threshold_ratio = 0
        pg.display.update()

    def handle(self, message_type, **kwargs):
        if message_type is MessageType.SOUND_RECEIVED:
            buffer = kwargs["data"]
            last_updated = kwargs["time"]
            self.draw_sound(buffer, last_updated)
        elif message_type is MessageType.ABOVE_THRESHOLD:
            ratios = kwargs["ratios"]
            self.last_threshold_ratios = ratios

    def draw_sound(self, buffer, last_updated):
        """Render new data on microphone channels
        """
        self.screen.fill(WHITE)

        dt = time.time() - last_updated
        plot_buffer = buffer[-int(RATE * 2):]
        for ch_idx, panel in enumerate(self.mic_panels):
            scale = 1 - np.exp(-self.last_threshold_ratios[ch_idx])  # TODO: make last_threshold_ratio per channel
            panel.set_data(plot_buffer[:, ch_idx], int(RATE * dt))
            panel.set_mic_level(scale)
            panel.draw(self.screen)

        pg.display.update()


class FileSaver(AppService):
    def __init__(self, app, folder, basename, min_duration=1.0):
        super(FileSaver, self).__init__(app)

        self.folder = folder
        self.basename = basename
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.min_duration = min_duration
        self.counter = 0

    def handle(self, message_type, **kwargs):
        if message_type is MessageType.RECORDING_CAPTURED:
            fs = kwargs["sampling_rate"]
            data = kwargs["data"].astype(np.int16)
            if len(data) / fs < self.min_duration:
                return
            else:
                print("saving wav")
                scipy.io.wavfile.write(
                    os.path.join(self.folder, "basename_{}_{}.wav".format(self.basename, self.counter)),
                    fs,
                    data# .astype(np.float32, order='C') / 32768.0
                )
                self.counter += 1


class RingBuffer(object):

    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.int16)
        self.last_updated = time.time()

    def capture(self, data):
        self.data[:-len(data)] = self.data[len(data):]
        self.data[-len(data):] = data
        self.last_updated = time.time()


class MicrophoneListener(AppService):

    def __init__(self, app, device=None):
        super(MicrophoneListener, self).__init__(app)
        self.device = device
        self.ringbuffer = RingBuffer((int(RATE * 20.0), self.app.channels))
        self.captured = 0
        self.stream = p.open(
            format=pyaudio.paInt16,
            input_device_index=2,
            channels=self.app.channels,
            rate=RATE,
            frames_per_buffer=CHUNK,
            input=True,
            output=False,
            stream_callback=self.callback
        )
        self.recording = False

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)[:]

        self.ringbuffer.capture(np.array([
            data[idx::self.app.channels] for idx in range(self.app.channels)
        ]).T)

        if self.recording is True:
            self.captured_since += len(data)

        return (in_data, pyaudio.paContinue)

    def handle(self, message_type, **kwargs):
        if message_type == MessageType.RECORDING:
            recording = kwargs["value"]
            if recording:
                self.recording_on()
            else:
                self.recording_off()
        elif message_type == MessageType.LOOP:
            if self.stream.is_active():
                self.emit(MessageType.SOUND_RECEIVED,
                    data=self.ringbuffer.data[:],
                    time=self.ringbuffer.last_updated
                )
        elif message_type == MessageType.LISTENING:
            on = kwargs["on"]
            if on is True:
                self.stream.start_stream()
            elif on is False:
                self.stream.stop_stream()
        elif message_type == MessageType.CLOSING:
            self.close()

    def recording_on(self):
        # refers to number of samples before onset to collect data
        self.captured_since = int(SILENT_BUFFER * RATE)
        print("\n recording *    ", end="\r")
        self.recording = True

    def recording_off(self):
        self.recording = False
        print("\n end recording *    ", end="\r")
        self.emit(
            MessageType.RECORDING_CAPTURED,
            data=self.ringbuffer.data[-self.captured_since:],
            sampling_rate=RATE,
        )
        print("Recorded {}s of data".format(self.captured_since / RATE))

    def close(self):
        self.stream.close()


# class App(object):
#     def __init__(self, channels=1):
#         self.channels = channels
#         self.clock = pg.time.Clock()
#         self.pubsub = PubSub()
#
#     def run(self):
#         try:
#             while True:
#                 for event in pg.event.get():
#                     if event.type == pg.QUIT:
#                         self.pubsub.emit(MessageType.CLOSING)
#                         pg.quit()
#                         sys.exit()
#
#                 self.pubsub.emit(MessageType.LOOP)
#                 self.clock.tick(60)  # 60 fps why not
#                 print(self.clock.get_fps())
#
#         except KeyboardInterrupt:
#             self.pubsub.emit(MessageType.CLOSING)
#             pg.quit()
#             sys.exit()

from PyQt5 import QtWidgets as widgets
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5 import QtGui as gui
import sys
# import ui_main
import numpy as np
import pyqtgraph


import sys
from PyQt5.QtWidgets import (QWidget, QLineEdit, QGridLayout,QLabel, QApplication)
import pyqtgraph

class Window(widgets.QWidget):
    def __init__(self, channels=1):
        super(Window, self).__init__()
        self.channels = channels
        self.edit = widgets.QLineEdit('TSLA', self)
        self.label = widgets.QLabel('-', self)
        self.guiplot = pyqtgraph.PlotWidget()
        self.guiplot.setYRange(-1000, 1000, padding=0)
        # self.guiplot.disableAutoRange()

        self.curve = None
        self.curve2 = None


        # self.guiplot.enableAutoRange()
        self.guiplot2 = pyqtgraph.PlotWidget()
        self.guiplot2.setYRange(-1000, 1000, padding=0)
        # self.guiplot2.setYRange(-1000, 1000, padding=0)
        # self.guiplot2.enableAutoRange()

        layout = QGridLayout(self)
        layout.addWidget(self.edit, 0, 0)
        layout.addWidget(self.label, 1, 0)
        layout.addWidget(self.guiplot, 2, 0, 3, 3)
        layout.addWidget(self.guiplot2, 5, 0, 3, 3)

        self.prevBar = None
        self.bars = None
        self.timer = QTimer()
        self.timer.start(1/60)
        self.timer.timeout.connect(self._loop)
        self.pubsub = PubSub()

    def _loop(self):
        self.pubsub.emit(MessageType.LOOP)

    def handle(self, message_type, **kwargs):
        if message_type is MessageType.SOUND_RECEIVED:
            buffer = kwargs["data"]
            last_updated = kwargs["time"]
            self.update(buffer, last_updated)

    def update(self, buffer, last_updated):
        dt = time.time() - last_updated
        plot_buffer = buffer[-int(RATE * 2)::10]
        # for ch_idx, panel in enumerate([0, 1]):
            # scale = 1 - np.exp(-self.last_threshold_ratios[ch_idx])  # TODO: make last_threshold_ratio per channel
        data = plot_buffer[:]
        # oint(RATE * dt)
        # panel.set_mic_level(scale)
        # panel.draw(self.screen)
        # self.guiplot.plot([0, len(data)], [200, 200], clear=True, color="red")

        data = bandpass_filter(data.T, RATE, 100, 10000).T
        if self.curve is None:
            self.curve = self.guiplot.plot(data[:, 0], clear=True)
        else:
            self.curve.setData(data[:, 0])

        if self.curve2 is None:
            self.curve2 = self.guiplot2.plot(data[:, 1], clear=True)
        else:
            self.curve2.setData(data[:, 1])



# Initialize pyaudio and pygame libraries
p = pyaudio.PyAudio()
pg.init()


def run():

    # Locate listening device?
    # info = p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')
    # for i in range(0, numdevices):
    #     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    # print(p.get_default_input_device_info())
    app2 = QApplication(sys.argv)

    window = Window(channels=2)
    window.show()

    # Setup services
    # gui = GUI(app)
    detector = SoundDetector(window)
    mics = MicrophoneListener(window)
    filesaver = FileSaver(
        window,
        "/auto/fhome/kevin/temp/stimuli",
        "birdy",
        min_duration=0.2 + 2 * SILENT_BUFFER,
    )

    # Subscribe
    window.pubsub.subscribe(MessageType.SOUND_RECEIVED, detector)
    # app.pubsub.subscribe(MessageType.SOUND_RECEIVED, gui)
    window.pubsub.subscribe(MessageType.SOUND_RECEIVED, window)

    # app.pubsub.subscribe(MessageType.ABOVE_THRESHOLD, gui)
    window.pubsub.subscribe(MessageType.RECORDING_CAPTURED, filesaver)
    window.pubsub.subscribe(MessageType.RECORDING, mics)
    window.pubsub.subscribe(MessageType.LISTENING, mics)
    window.pubsub.subscribe(MessageType.LOOP, mics)
    window.pubsub.subscribe(MessageType.CLOSING, mics)

    sys.exit(app2.exec_())

    p.terminate()



if __name__ == "__main__":
    run()
