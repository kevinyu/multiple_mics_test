"""Interface independent application controller
"""

from core.listen import Microphone, list_devices
from core.streams import IndependentStreamController, SynchronizedStreamController


class AppController(object):
    """Configurable abstract recording app on a single device

    Methods
    =======
    apply_config(config)
    set_synchronized(bool)
    run()
    stop()

    The following methods all basically pass through to the MultiStreamController

        set_stream_gain(self, stream_index: int, gain: float)
        set_stream_name(self, stream_index: int, name: str)
        add_stream(self, **config)
        remove_stream(self, stream_index: int)

    Attributes
    ==========
    mic: core.listen.Microphone
    stream_controller: core.streams.MultiStreamController
    """

    def __init__(self):
        # The mic must be instantiated within the main asyncio loop coroutine of the relevant app!
        self.mic = None
        self._config = {}
        self.stream_controller = None

    def list_devices(self):
        return list_devices()

    def apply_config(self, config: dict):
        """Apply a configuration dictionary to connect microphone and streams

        TODO: config documentation
        """
        self.stop()

        self._config = config
        device_name = config.get("device_name")
        streams = config.get("streams", [])

        # Apply defaults from the op layer of the config to each stream
        for stream in streams:
            if "gain" not in stream:
                stream["gain"] = config.get("gain", 0)
            for key, val in config.get("save", {}).items():
                if key not in stream.get("save", {}):
                    if "save" not in stream:
                        stream["save"] = {}
                    stream["save"][key] = val
            for key, val in config.get("detect", {}).items():
                if key not in stream.get("detect", {}):
                    if "detect" not in stream:
                        stream["detect"] = {}
                    stream["detect"][key] = val

        config["streams"] = streams

        self.mic = Microphone(device_name=device_name)

        if config["synchronized"] is True:
            self.stream_controller = SynchronizedStreamController(mic=self.mic)
        else:
            self.stream_controller = IndependentStreamController(mic=self.mic)
        self.stream_controller.apply_config(config)

    def set_synchronized(self, synchronized: bool):
        """Reapply the current config with a new synchronization setting"""
        current_synchronized = bool(self._config.get("synchronized"))
        if synchronized != current_synchronized:
            self.stream_controller.stop()
            del self.stream_controller
            self._config["synchronized"] = synchronized
            self.apply_config(self._config)

    def get_streams(self):
        return self.stream_controller.get_streams()

    def set_stream_gain(self, stream_index: int, gain: float):
        self.stream_controller.set_stream_gain(gain)

    def set_stream_name(self, stream_index: int, name: str):
        self.stream_controller.set_stream_name(name)

    def add_stream(self, **config):
        self.stream_controller.add_stream(**config)

    def remove_stream(self, stream_index: int):
        self.stream_controller.remove_stream(stream_index)

    def run(self):
        if not self.mic:
            raise RuntimeError("Cannot run App without Microphone instantiated. May need to apply_config() first.")
        return self.mic.run()

    def stop(self):
        if self.mic:
            self.mic.stop()
