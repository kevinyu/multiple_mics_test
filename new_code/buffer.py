"""Implementation of a circular buffer with similar api to collections.deque
"""
import numpy as np


class Buffer(object):
    def __init__(self, maxlen=0):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0

        self.overlapping = False
        self._write_at = 0
        self._start_pointer = 0
        self.data = None

    def __len__(self):
        return self.length

    def __array__(self):
        if self.data is None:
            return np.array([])
        else:
            part1 = self.data[self._start_pointer:self._start_pointer + self.length]
            if self._start_pointer + self.length > self.maxlen:
                part2 = self.data[:self.length - part1.shape[0]]
                return np.vstack([part1, part2])
            else:
                return part1

    def clear(self):
        self.data = None
        self.length = 0
        self._write_at = 0
        self._start_pointer = 0
        self.overlapping = False

    def extend(self, data):
        if self.maxlen == 0:
            return

        to_add = np.array(data)
        if to_add.ndim == 1:
            to_add = to_add[:, None]

        if self.data is None:
            self.data = np.zeros((self.maxlen, to_add.shape[1]))
            self.length = 0
            self._write_at = 0
            self._start_pointer = 0
            self.overlapping = False

        # Validate that the dimensions are correct
        if self.data.shape[1] != to_add.shape[1]:
            raise ValueError("Cannot extend array with incompatible shape")

        if len(to_add) > len(self.data):
            self.data = to_add[-len(self.data):]
            self.length = 0
            self._write_at = 0
            self._start_pointer = 0
            self.overlapping = False
            self.length = self.maxlen
        elif self._write_at + len(to_add) <= self.maxlen:
            self.data[self._write_at:self._write_at + len(to_add)] = to_add
            self._write_at += len(to_add)
            self.length = self._write_at
            if self.overlapping:
                self._start_pointer = self._write_at
            else:
                self._start_pointer = 0
        else:
            self.overlapping = True
            first_part_size = self.maxlen - self._write_at
            second_part_size = len(to_add) - first_part_size
            self.data[self._write_at:] = to_add[:first_part_size]
            self.data[:second_part_size] = to_add[first_part_size:]
            self._write_at = second_part_size
            self._start_pointer = self._write_at
            self.length = self.maxlen


class Busffer(object):
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
