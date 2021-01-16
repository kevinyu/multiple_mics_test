import os
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
        print(self.to_string(size.columns - 1), end="\r")
