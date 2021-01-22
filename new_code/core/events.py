import logging
import weakref

from configure_logging import handler


logger = logging.getLogger(__name__)
logger.addHandler(handler)


class Signal(object):
    def __init__(self, debug=False):
        self._debug = debug
        self.callbacks = {}

    def emit(self, *args, **kwargs):
        """Send an event to Signal's registered callbacks"""
        for id_ in self.callbacks:
            # callback = self.callbacks[id_]()
            callback = self.callbacks[id_]
            if self._debug:
                print("DEBUG Signal: {} {} {}".format(callback, args, kwargs))
            if callback is not None:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error("Callback {} failed with {}: arguments {} {}".format(
                        callback, e, args, kwargs
                    ))
                    raise

    def connect(self, callback):
        """Register a callback to be called when signal is emitted"""
        id_ = id(callback)
        # self.callbacks[id_] = weakref.ref(callback)
        self.callbacks[id_] = callback

    def disconnect(self, callback):
        id_ = id(callback)
        if id_ in self.callbacks:
            del self.callbacks[id_]


class AsyncSignal(Signal):
    def __init__(self, debug=False):
        self._debug = debug
        self.callbacks = {}

    async def emit(self, *args, **kwargs):
        """Send an event to Signal's registered callbacks"""
        for id_ in self.callbacks:
            # callback = self.callbacks[id_]()
            callback = self.callbacks[id_]
            if self._debug:
                print("DEBUG Signal: {} {} {}".format(callback, args, kwargs))
            if callback is not None:
                try:
                    callback(*args, **kwargs)
                except:
                    raise
