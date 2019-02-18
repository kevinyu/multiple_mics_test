"""
PyAudio Example: Make a wire between input and output (i.e., record a
few samples and play them back immediately).
"""
import os
import sys
import time
from threading import Thread

import numpy as np
import pyaudio
import pygame as pg
import scipy.io.wavfile


CHUNK = 1024 * 4
WIDTH = 2
CHANNELS = 1
RATE = 30000
SILENT_BUFFER = 1.0
CROSSINGS_THRESHOLD = 0.2 * 200 # (heuristic: 200 crossings per second during sound)
GUI_WIDTH = 640
GUI_HEIGHT = 480
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
                amp_threshold=200,
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
        detection_window = buffer[-self.detection_window:, 0]
        threshold_crossings = np.nonzero(
            np.diff(detection_window > self.amp_threshold)
        )[0]

        self.emit(
            MessageType.ABOVE_THRESHOLD,
            ratio=int(threshold_crossings.size) / CROSSINGS_THRESHOLD
        )

        return int(threshold_crossings.size) > CROSSINGS_THRESHOLD


class GUI(AppService):

    def __init__(self, app):
        super(GUI, self).__init__(app)
        self.screen = pg.display.set_mode((GUI_WIDTH, GUI_HEIGHT), 0, 32)
        pg.display.set_caption("Live Mic Recording")
        self.screen.fill(WHITE)
        self.last_threshold_ratio = 0

    def handle(self, message_type, **kwargs):
        if message_type is MessageType.SOUND_RECEIVED:
            buffer = kwargs["data"]
            self.draw_sound(buffer)
        elif message_type is MessageType.ABOVE_THRESHOLD:
            ratio = kwargs["ratio"]
            self.last_threshold_ratio = ratio

    def draw_sound(self, buffer):
        """Render new data on microphone channels
        """
        plot_buffer = buffer[-int(RATE * 2):, 0]

        # Draw a red circle to indicate how the sound compares to threshold
        scale = 1 - np.exp(-self.last_threshold_ratio)
        radius = 20 + int(scale * 30)

        self.screen.fill(WHITE)
        pg.draw.circle(
            self.screen,
            (200, 100, 100) if self.last_threshold_ratio > 0.5 else BLACK,
            (100, 100),
            radius
        )
        pg.draw.circle(self.screen, BLACK, (100, 100), 30)

        # Plot the sound pressure waveform
        line = pg.draw.aalines(
            self.screen,
            BLACK,
            False,
            list(zip(*(np.linspace(160, 550, len(plot_buffer)), 100 + plot_buffer / 100.0)))
        )

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
                    data.flatten()# .astype(np.float32, order='C') / 32768.0
                )
                self.counter += 1


class RingBuffer(object):

    def __init__(self, shape):
        self.data = np.zeros(shape, dtype=np.int16)

    def capture(self, data):
        self.data[:-len(data)] = self.data[len(data):]
        self.data[-len(data):] = data


class MicrophoneListener(AppService):

    def __init__(self, app, device=None):
        super(MicrophoneListener, self).__init__(app)
        self.device = device
        self.ringbuffer = RingBuffer((int(RATE * 20.0), CHANNELS))
        self.captured = 0
        self.stream = p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=RATE,
            frames_per_buffer=CHUNK,
            input=True,
            output=False,
            stream_callback=self.callback
        )
        self.recording = False

    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        if data.ndim == 1:
            data = data[:, None]

        self.ringbuffer.capture(data)

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
                self.emit(MessageType.SOUND_RECEIVED, data=self.ringbuffer.data)
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
        print(" recording *    ", end="\r")
        self.recording = True

    def recording_off(self):
        self.recording = False
        print(" end recording *    ", end="\r")
        self.emit(
            MessageType.RECORDING_CAPTURED,
            data=self.ringbuffer.data[-self.captured_since:],
            sampling_rate=RATE,
        )
        print("Recorded {}s of data".format(self.captured_since / RATE))

    def close(self):
        self.stream.close()


class App(object):
    def __init__(self):
        self.clock = pg.time.Clock()
        self.pubsub = PubSub()

    def run(self):
        try:
            while True:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.pubsub.emit(MessageType.CLOSING)
                        pg.quit()
                        sys.exit()

                self.pubsub.emit(MessageType.LOOP)
                self.clock.tick(60)  # 60 fps why not
        except KeyboardInterrupt:
            self.pubsub.emit(MessageType.CLOSING)
            pg.quit()
            sys.exit()


if __name__ == "__main__":
    app = App()

    # Initialize pyaudio and pygame libraries
    p = pyaudio.PyAudio()
    pg.init()

    # Locate listening device?
    # info = p.get_host_api_info_by_index(0)
    # numdevices = info.get('deviceCount')
    # for i in range(0, numdevices):
    #     if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #         print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
    # print(p.get_default_input_device_info())

    # Setup services
    gui = GUI(app)
    detector = SoundDetector(app)
    mics = MicrophoneListener(app)
    filesaver = FileSaver(
        app,
        "/Users/kevinyu/Projects/bg_audio/temp", "birdy",
        min_duration=0.2 + 2 * SILENT_BUFFER,
    )

    # Subscribe
    app.pubsub.subscribe(MessageType.SOUND_RECEIVED, detector)
    app.pubsub.subscribe(MessageType.SOUND_RECEIVED, gui)
    app.pubsub.subscribe(MessageType.ABOVE_THRESHOLD, gui)
    app.pubsub.subscribe(MessageType.RECORDING_CAPTURED, filesaver)
    app.pubsub.subscribe(MessageType.RECORDING, mics)
    app.pubsub.subscribe(MessageType.LISTENING, mics)
    app.pubsub.subscribe(MessageType.LOOP, mics)
    app.pubsub.subscribe(MessageType.CLOSING, mics)

    app.run()

    p.terminate()
