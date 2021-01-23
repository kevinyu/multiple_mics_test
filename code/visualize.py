import os
import time
from collections.abc import Iterable


class Powerbar(object):

    parts = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]

    def __init__(self, max_value=1, channels=1):
        self.max_value = max_value
        self.channel_values = {}
        self.set_channels(channels)

    def set_max(self, max_value):
        self.max_value = max_value

    def set_channels(self, channels):
        try:
            channels = int(channels)
        except:
            pass

        if isinstance(channels, int):
            self.channels = range(channels)
        else:
            self.channels = channels
        self.channel_values = dict((ch, 0) for ch in self.channels)

    def set_channel_value(self, channel, value):
        self.channel_values[channel] = value

    def channel_to_string(self, channel, width):
        value = self.channel_values[channel]
        chars_per_value = width / self.max_value
        if value >= self.max_value:
            n_full_bars = width
            return self.parts[-1] * n_full_bars
        else:
            n_full_bars = int(value * chars_per_value)
            leftover_value = value - (n_full_bars / chars_per_value)
            leftover_idx = int(len(self.parts) * leftover_value * chars_per_value)
            return "{}{}".format(self.parts[-1] * n_full_bars, self.parts[leftover_idx]).ljust(width)

    def to_string(self, width):
        width_per_channel = width // len(self.channels)
        if width_per_channel == 0:
            raise RuntimeError("Not enough room in {} to render {} powerbars".format(
                width,
                len(self.channels)
            ))

        output = ""
        for channel in self.channels:
            output += self.channel_to_string(channel, width_per_channel)

        return output

    def print(self, end="\r"):
        size = os.get_terminal_size()
        print(" " + self.to_string(size.columns - 2), end="\r")


class DetectionsPowerbar(Powerbar):
    """Displays mic signals on a single line as well as a detection indicator

    The detection indicators turn on when set_detected(channel) is called
    and decay after a pre-specified amount of time. This time is
    controlled by the event loop but can also be run synchronously.
    """
    detection_symbols = {True: "⬤", False: "◯"}

    def __init__(self, decay_time, *args, **kwargs):
        self._detections = {}
        self.decay_time = decay_time
        super().__init__(*args, **kwargs)

    def set_detected(self, channel, *args, **kwargs):
        now = time.time()
        if channel is None:
            for ch in self._detections:
                self._detections[ch] = now
        else:
            self._detections[channel] = now

    def set_channels(self, channels):
        super().set_channels(channels)
        self._detections = dict((ch, None) for ch in self.channels)

    def channel_to_string(self, channel, width):
        bar_output = super().channel_to_string(channel, width - 3)

        last_detection = self._detections.get(channel)
        is_detected = (
            last_detection is not None
            and time.time() - last_detection < self.decay_time
        )

        return " {} {}".format(self.detection_symbols[is_detected], bar_output)
