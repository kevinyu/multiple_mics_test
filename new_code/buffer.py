"""Implementation of a circular buffer with similar api to collections.deque
"""
import numpy as np


class Buffer(object):
    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = None

    def __len__(self):
        return self.length

    def __array__(self):
        if self.data is None:
            return np.array([])

        else:
            return self.data[-self.length:]

    def clear(self):
        self.data = None
        self.length = 0

    def extend(self, data):
        to_add = np.array(data)
        if to_add.ndim == 1:
            to_add = to_add[:, None]

        if self.data is None:
            if self.maxlen:
                self.data = np.zeros((self.maxlen, to_add.shape[1]))
                self.length = 0
            else:
                self.data = np.zeros((0, to_add.shape[1]))
                self.length = 0

        # Validate that the dimensions are correct
        if self.data.shape[1] != to_add.shape[1]:
            raise ValueError("Cannot extend array with incompatible shape")

        if self.maxlen:
            if len(data) >= self.maxlen:
                self.data[:] = to_add[:-self.maxlen:]
                self.length = len(self.data)
            else:
                self.data = np.roll(self.data, -len(to_add))
                self.data[-len(to_add):] = to_add
                self.length = min(self.maxlen, self.length + len(to_add))
        else:
            self.data = np.concatenate([self.data, to_add])
            self.length = len(self.data)
