from PyQt5.QtCore import QSettings
import numpy as np


qsettings = QSettings("Theunissen Lab", "TLabRecorder")


class Settings(object):
    CHUNK = 1024
    RATE = 44100
    DTYPE = np.int16

    GAIN = np.array([0])

    USE_SOUNDDEVICE = True
    CHANNEL_NAMES = {}

    BASE_DIRECTORY = None
    FILENAME_SUFFIX = "time"
    SAVE_DIRECTORY = None
    SAVE_CONTINUOUSLY = False
    FILE_DURATION = 30      # seconds
    MIN_FILE_DURATION = 1.0   # seconds
    MAX_TRIGGERED_DURATION = 20   # seconds

    DETECTION_WINDOW = 0.1  # seconds
    DETECTION_BUFFER = 0.3  # seconds
    MIN_POWER_THRESHOLD = 1
    DEFAULT_POWER_THRESHOLD = 20
    MAX_POWER_THRESHOLD = 100
    DETECTION_CROSSINGS_PER_CHUNK = 20

    PLOT_DURATION = 5.0

    @classmethod
    def get(cls, key, otherwise=None):
        return qsettings.value(key, getattr(cls, key, otherwise))

    @classmethod
    def set(cls, key, val):
        qsettings.setValue(key, val)
        qsettings.sync()
        setattr(cls, key, val)
        

