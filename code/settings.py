from PyQt5.QtCore import QSettings
import numpy as np


qsettings = QSettings("Theunissen Lab", "TLabRecorder")


class Settings(object):
    CHUNK = 1024
    RATE = 44100
    DTYPE = np.int16

    DEFAULT_GAIN = 20
    GAIN = np.array([DEFAULT_GAIN])

    USE_SOUNDDEVICE = True
    CHANNEL_NAMES = {}
    SAVE_CHANNELS_SEPARATELY = True

    BASE_DIRECTORY = None
    FILENAME_SUFFIX = "time"
    SAVE_DIRECTORY = None
    SAVE_CONTINUOUSLY = False
    FILE_DURATION = 10      # seconds
    MIN_FILE_DURATION = 1.0   # seconds
    MAX_TRIGGERED_DURATION = 20   # seconds

    DISPLAY_AMP_DOWNSAMPLE = 20

    FRAMERATE = 20.0
    SPEC_LEVELS_MIN = -10
    SPEC_LEVELS_MAX = 150
    DEFAULT_POWER_THRESHOLD = 500
    MAX_POWER_THRESHOLD = 5000
    DETECTION_CROSSINGS_PER_CHUNK = 20

    DETECTION_WINDOW = 0.1  # seconds
    DETECTION_BUFFER = 0.3  # seconds
    MIN_POWER_THRESHOLD = 1

    PLOT_DURATION = 4.0

    @classmethod
    def get(cls, key, otherwise=None):
        return qsettings.value(key, getattr(cls, key, otherwise))

    @classmethod
    def set(cls, key, val):
        qsettings.setValue(key, val)
        qsettings.sync()
        setattr(cls, key, val)
