import collections
import logging

import numpy as np


logger = logging.getLogger(__name__)


class MicrophoneListener(object):
    def receive_data(self, data):
        logger.info("{} received data of size {}".format(
            self, data.shape))


class SoundDetector(MicrophoneListener):
    def __init__(self, size):
        self._buffer = collections.deque(maxlen=size)

    def receive_data(self, data):
        self._buffer.extend(data)


class SoundSaver(MicrophoneListener):
    def __init__(self, size):
        """
        Parameters
        ----------
        size : int
            Size of each file to be saved in samples
        """
        self._buffer = collections.deque()
        self._idx = 0
        self.size = size

    def receive_data(self, data):
        self._buffer.extend(data)
        if len(self._buffer) > self.size:
            data = np.array(self._buffer)
            self._save(data[:self.size])
            self._buffer.clear()
            self._buffer.extend(data[self.size:])

    def _save(self, data):
        print("saving", data.shape)
