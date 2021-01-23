import unittest
from unittest import mock

import numpy as np

from events import Signal
from ringbuffer import RingBuffer


class TestSignal(unittest.TestCase):

    def test_signal_callbacks_reference(self):
        cb = mock.Mock()

        signal = Signal()
        signal.connect(cb)
        cb_id = id(cb)

        # Test that the callback is registered
        self.assertEqual(len(signal.callbacks), 1)
        # This line is Truer when using weakref...
        # self.assertIs(signal.callbacks[cb_id](), cb)
        self.assertIs(signal.callbacks[cb_id], cb)

    def test_signal_emit(self):
        cb1 = mock.Mock()
        cb2 = mock.Mock()

        signal = Signal()
        signal.connect(cb1)
        signal.connect(cb2)

        # Test that emit calls both call backs
        signal.emit(10, a=20)
        cb1.assert_called_with(10, a=20)
        cb2.assert_called_with(10, a=20)

    def test_signal_disconnect(self):
        cb1 = mock.Mock()
        cb2 = mock.Mock()

        signal = Signal()
        signal.connect(cb1)
        signal.connect(cb2)

        self.assertEqual(len(signal.callbacks), 2)

        # Test disconnect
        signal.disconnect(cb1)
        self.assertEqual(len(signal.callbacks), 1)
        self.assertNotIn(id(cb1), signal.callbacks)
        self.assertIn(id(cb2), signal.callbacks)

        # Test emit not sent to disconnected
        signal.emit(1)
        cb2.assert_called_with(1)
        self.assertEqual(cb1.call_count, 0)


class TestRingBuffer(unittest.TestCase):

    def test_buffer_init(self):
        b = RingBuffer()
        self.assertEqual(b.maxlen, 0)
        self.assertIs(b.n_channels, None)
        self.assertIs(b._init_n_channels, None)
        self.assertEqual(b._ringbuffer.shape, (0, 0))

        b = RingBuffer(maxlen=10)
        self.assertEqual(b.maxlen, 10)
        self.assertIs(b.n_channels, None)
        self.assertIs(b._init_n_channels, None)
        self.assertEqual(b._ringbuffer.shape, (10, 0))

        b = RingBuffer(maxlen=10, n_channels=2)
        self.assertEqual(b.maxlen, 10)
        self.assertEqual(b.n_channels, 2)
        self.assertEqual(b._init_n_channels, 2)
        self.assertEqual(b._ringbuffer.shape, (10, 2))

        b = RingBuffer(n_channels=3)
        self.assertEqual(b.maxlen, 0)
        self.assertEqual(b.n_channels, 3)
        self.assertEqual(b._init_n_channels, 3)
        self.assertEqual(b._ringbuffer.shape, (0, 3))

    def test_buffer_infer_channels(self):
        """Test that the RingBuffer infers channels when initialized without n_channels"""
        b = RingBuffer(maxlen=10)
        self.assertIs(b.n_channels, None)
        self.assertIs(b._init_n_channels, None)

        b.extend(np.ones((5, 3)))
        self.assertIs(b.n_channels, 3, "Extend on RingBuffer with channels=None should infer number of channels.")
        self.assertIs(b._init_n_channels, None, "Inferred channels should not overwrite original channel param.")

        with self.assertRaises(ValueError):
            b.extend(np.ones((3, 2)))

        b.clear()
        self.assertIs(b.n_channels, None)
        self.assertIs(b._init_n_channels, None)

    def test_buffer_extend(self):
        """Test the RingBuffer extend method behaves normally"""
        b = RingBuffer(maxlen=10, n_channels=2)
        np.testing.assert_array_equal(b._ringbuffer, np.zeros((10, 2)))

        b.extend(np.ones((6, 2)))
        np.testing.assert_array_equal(b._ringbuffer[:6], np.ones((6, 2)))
        np.testing.assert_array_equal(b._ringbuffer[6:], np.zeros((4, 2)))
        self.assertEqual(b._length, 6)
        self.assertEqual(b._write_at, 6)
        self.assertEqual(b._start, 0)
        self.assertFalse(b._overlapping)

        np.testing.assert_array_equal(
            b.to_array(),
            np.vstack([np.ones((6, 2))])
        )

        # Testing when buffer circles back around
        b.extend(2 * np.ones((6, 2)))
        np.testing.assert_array_equal(b._ringbuffer[:2], 2 * np.ones((2, 2)))
        np.testing.assert_array_equal(b._ringbuffer[2:6], np.ones((4, 2)))
        np.testing.assert_array_equal(b._ringbuffer[6:], 2 * np.ones((4, 2)))
        self.assertEqual(b._length, 10)
        self.assertEqual(b._write_at, 2)
        self.assertEqual(b._start, 2)
        self.assertTrue(b._overlapping)

        np.testing.assert_array_equal(
            b.to_array(),
            np.vstack([np.ones((4, 2)), 2 * np.ones((6, 2))])
        )

        # Test a normal append while overlapping is True
        b.extend(3 * np.ones((7, 2)))
        np.testing.assert_array_equal(b._ringbuffer[:2], 2 * np.ones((2, 2)))
        np.testing.assert_array_equal(b._ringbuffer[2:9], 3 * np.ones((7, 2)))
        np.testing.assert_array_equal(b._ringbuffer[9:], 2 * np.ones((1, 2)))

        self.assertEqual(b._length, 10)
        self.assertEqual(b._write_at, 9)
        self.assertEqual(b._start, 9)
        self.assertTrue(b._overlapping)

        np.testing.assert_array_equal(
            b.to_array(),
            np.vstack([2 * np.ones((3, 2)), 3 * np.ones((7, 2))])
        )

        # Test a append that circles back around while overlapping is already true
        b.extend(4 * np.ones((2, 2)))
        np.testing.assert_array_equal(b._ringbuffer[:1], 4 * np.ones((1, 2)))
        np.testing.assert_array_equal(b._ringbuffer[1:2], 2 * np.ones((1, 2)))
        np.testing.assert_array_equal(b._ringbuffer[2:9], 3 * np.ones((7, 2)))
        np.testing.assert_array_equal(b._ringbuffer[9:], 4 * np.ones((1, 2)))
        self.assertEqual(b._length, 10)
        self.assertEqual(b._write_at, 1)
        self.assertEqual(b._start, 1)
        self.assertTrue(b._overlapping)

        np.testing.assert_array_equal(
            b.to_array(),
            np.vstack([2 * np.ones((1, 2)), 3 * np.ones((7, 2)), 4 * np.ones((2, 2))])
        )

    def test_buffer_extend_longer_than_maxlen(self):
        """Test RingBuffer.extend() for data longer than the buffer itself"""
        b = RingBuffer(maxlen=5)

        data = np.arange(6)[:, None]

        b.extend(data)
        np.testing.assert_array_equal(b._ringbuffer, data[1:])
        np.testing.assert_array_equal(b.to_array(), data[1:])

    def test_buffer_extend_exactly_to_end(self):
        """Test RingBuffer.extend() for data that reaches the end of the buffer exactly"""
        b = RingBuffer(maxlen=5)
        b.extend(np.ones((3, 1)))
        b.extend(np.ones((2, 1)))

        np.testing.assert_array_equal(b._ringbuffer, np.ones((5, 1)))
        np.testing.assert_array_equal(b.to_array(), np.ones((5, 1)))
        self.assertEqual(b._length, 5)
        self.assertEqual(b._write_at, 0)
        self.assertEqual(b._start, 0)
        self.assertTrue(b._overlapping)

    def test_buffer_clear_resets(self):
        b = RingBuffer(maxlen=10)
        b.extend(np.ones((9, 1)))
        b.extend(np.ones((6, 1)))

        b.clear()

        self.assertEqual(b.maxlen, 10)
        self.assertIs(b.n_channels, None)
        self.assertIs(b._init_n_channels, None)
        self.assertEqual(b._ringbuffer.shape, (10, 1), "The underlying ringbuffer array should not be affected until next write")

        b = RingBuffer(maxlen=10, n_channels=3)
        b.extend(np.ones((9, 3)))
        b.extend(np.ones((6, 3)))

        b.clear()

        self.assertEqual(b.maxlen, 10)
        self.assertIs(b.n_channels, 3)
        self.assertIs(b._init_n_channels, 3)
        self.assertEqual(b._ringbuffer.shape, (10, 3))

    def test_buffer_new_shape_after_clear(self):
        """Test that arrays with a different number of channels can be extended after clearing"""
        b = RingBuffer(maxlen=10)
        self.assertEqual(b._ringbuffer.shape[1], 0)

        b.extend(np.ones((4, 2)))
        self.assertEqual(b._ringbuffer.shape[1], 2)

        with self.assertRaises(ValueError):
            # Mismatched channel numbers
            b.extend(np.ones((4, 4)))

        b.clear()
        b.extend(np.ones((4, 4)))
        self.assertEqual(b._ringbuffer.shape[1], 4)


