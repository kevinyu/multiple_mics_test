"""Interface independent application controller
"""

from core.config import RecordingConfig
from core.listen import Microphone, list_devices
from core.streams import IndependentStreamController, SynchronizedStreamController


class AppController(object):
    """Headless recording app

    Methods
    =======
    list_devices()
    apply_config(config)

    Controls
    ========
    run()
    stop()
    set_synchronized(bool)

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

    _public_stream_controller_methods = (
        "get_streams",
        "set_save_config",
        "set_stream_name",
        "set_stream_channel",
        "set_monitor",
        "set_stream_monitor",
        "set_gain",
        "set_stream_gain",
        "set_threshold",
        "set_recording_mode",
        "set_stream_recording_mode",
        "set_stream_threshold",
        "apply_collector_config",
        "get_buffer_duration",
        "add_stream",
        "remove_stream",
    )

    def __getattr__(self, name):
        if name in self._public_stream_controller_methods:
            return getattr(self.stream_controller, name)
        else:
            return super().__getattr__(name)

    def __init__(self):
        # The mic must be instantiated within the main asyncio loop coroutine of the relevant app!
        super().__init__()
        self.mic = None
        self._config = RecordingConfig()
        self.stream_controller = None

    def list_devices(self):
        # This doesn't really belong on here but its nice to have easy access to this function
        return list_devices()

    def apply_config(self, config: dict):
        """Apply a configuration dictionary to connect microphone and streams
        """
        self.stop()

        self._config = RecordingConfig(config)

        device_name = config.get("device_name")
        streams = config.get("streams", [])

        self.mic = Microphone(device_name=device_name)

        if config["synchronized"] is True:
            self.stream_controller = SynchronizedStreamController(mic=self.mic)
        else:
            self.stream_controller = IndependentStreamController(mic=self.mic)

        self.stream_controller.apply_config(self._config.inherited())

    def to_config(self):
        """Convert the current app state into a configuration dictionary

        This dictionary can be used to repopulate the app via apply_config()
        """
        config = {}
        config["device_name"] = self.mic.device_name
        config["synchronized"] = isinstance(self.stream_controller, SynchronizedStreamController)
        config["save"] = self.stream_controller.get_save_config()
        config["gain"] = self._config.get("gain", 0)
        stream_controller_config = self.stream_controller.to_config()
        config.update(stream_controller_config)
        return RecordingConfig(config).deherited()

    def save_config(self, path):
        self.to_config().to_json(path)

    def set_synchronized(self, synchronized: bool):
        """Reapply the current config with a new synchronization setting"""
        current_config = self.to_config()
        current_synchronized = bool(current_config.get("synchronized"))
        if synchronized != current_synchronized:
            self.stream_controller.stop()
            del self.stream_controller
            current_config["synchronized"] = synchronized
            self.apply_config(current_config)

    def run(self):
        if not self.mic:
            raise RuntimeError("Cannot run App without Microphone instantiated. May need to apply_config() first.")
        return self.mic.run()

    def stop(self):
        if self.mic:
            self.mic.stop()
