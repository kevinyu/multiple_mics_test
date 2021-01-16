import unittest
from unittest import mock
from events import Signal


class TestSignal(unittest.TestCase):

    def test_signal_callbacks_reference(self):
        cb = mock.Mock()

        signal = Signal()
        signal.connect(cb)
        cb_id = id(cb)

        # Test that the callback is registered
        self.assertEqual(len(signal.callbacks), 1)
        self.assertIs(signal.callbacks[cb_id](), cb)

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
