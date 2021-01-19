"""Implementation of a circular buffer with similar api to collections.deque
"""
import numpy as np


class RingBuffer(object):
    """A circular buffer

    Allocates space for the max buffer length. Use RingBuffer.extend(data)
    to add to the buffer and RingBuffer.to_array() to get the current contents.

    Methods
    =======
    RingBuffer.extend(data)
        Extends the buffer with a 2D numpy array
    RingBuffer.to_array()
        Return an array representation of data in the buffer (copied
        so that it can be modified without affecting the buffer)
    RingBuffer.read_last(n_samples)
        Read the last n_samples that were stored in the buffer.
    RingBuffer.clear()
        Reset the ring buffer
    """

    DEFAULT_DTYPE = np.int16

    def __init__(self, maxlen=0, n_channels=None, dtype=None):
        """Initialize circular buffer

        Params
        ======
        maxlen : int (default 0)
            Maximum size of buffer
        n_channels : int (default None)
            Enforce number of channels in buffer. If None,
            will choose the number of channels the first time .extend()
            is called.
        dtype : type (default None)
            Enforce datatype of buffer. If None,
            will choose the datatype the first time .extend()
            is called.
        """
        self.maxlen = maxlen

        # Keep track of original value for if the buffer is cleared
        self._init_n_channels = n_channels
        self.n_channels = n_channels
        self._init_dtype = dtype
        self.dtype = dtype

        # Data is stored in a numpy array of maxlen even when
        # the amount of data is smaller than that. When data
        # exceeds maxlen we loop around and keep track of where we
        # started.
        self._write_at = 0  # Where the next data should be written
        self._length = 0  # The amount of samples of real data in the buffer
        self._start = 0  # Starting index where data should be read from
        self._overlapping = False  # Has the data wrapper around the end
        self._ringbuffer = np.zeros((self.maxlen, self.n_channels or 0), dtype=self.dtype or self.DEFAULT_DTYPE)

    def __len__(self):
        return self._length

    def __array__(self):
        return self.to_array()

    def to_array(self):
        # Read to the end and then wrap around to the beginning
        # if self._start + self._length > self.maxlen:
        if self._overlapping:
            return np.roll(self._ringbuffer, -self._start, axis=0)
        else:
            return self._ringbuffer[:self._length].copy()

    def clear(self):
        self._write_at = 0
        self._length = 0
        self._start = 0
        self.n_channels = self._init_n_channels
        self.dtype = self._init_dtype
        self._overlapping = False

    def extend(self, data):
        """Extend the buffer with a 2D (samples x channels) array

        Requires shape to be consistent with existing data
        """
        if self.maxlen == 0:
            return

        # Reshape 1-D signals to be 2D with one channel
        to_add = np.array(data)
        if self.dtype is None:
            self.dtype = to_add.dtype
            self._ringbuffer = self._ringbuffer.astype(self.dtype)

        if to_add.ndim == 1:
            to_add = to_add[:, None]

        # Enforce channels here
        if self.n_channels and to_add.shape[1] != self.n_channels:
            raise ValueError("Cannot extend {} channel Buffer with data of shape {}".format(
                self.n_channels,
                to_add.shape
            ))

        if self._length == 0 and self.n_channels is None:
            self._ringbuffer = np.zeros((self.maxlen, to_add.shape[1]), dtype=self.dtype)
            self.n_channels = to_add.shape[1]

        if len(to_add) > self.maxlen:
            self._ringbuffer[:] = to_add[-self.maxlen:]
            self._write_at = 0
            self._length = self.maxlen
            self._start = 0
            self._overlapping = False
        elif self._write_at + len(to_add) < self.maxlen:
            self._ringbuffer[self._write_at:self._write_at + len(to_add)] = to_add
            self._write_at += len(to_add)
            self._length = self.maxlen if self._overlapping else self._write_at
            self._start = self._write_at if self._overlapping else 0
        else:
            first_part_size = self.maxlen - self._write_at
            first_part = to_add[:first_part_size]
            second_part = to_add[first_part_size:]
            self._ringbuffer[self._write_at:] = first_part
            self._ringbuffer[:len(second_part)] = second_part
            self._write_at = len(second_part)
            self._length = self.maxlen
            self._start = self._write_at
            self._overlapping = True

    def read_last(self, n_samples):
        """Reads the last n_samples that were stored in the buffer"""
        return self.to_array()[-n_samples:]
