import collections
import matplotlib.cm as cm

import pyqtgraph as pg
import numpy as np

from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot

from core.config import PlotConfig


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

    DISPLAY_AMP_DOWNSAMPLE = 50

    FRAMERATE = 20.0
    SPEC_LEVELS_MIN = -10
    SPEC_LEVELS_MAX = 150
    DEFAULT_POWER_THRESHOLD = 500
    MAX_POWER_THRESHOLD = 9999
    DETECTION_CROSSINGS_PER_CHUNK = 20

    DETECTION_WINDOW = 0.1  # seconds
    DETECTION_BUFFER = 0.3  # seconds
    MIN_POWER_THRESHOLD = 1

    PLOT_DURATION = 4.0


def cmapToColormap(cmap, nTicks=16):
    """
    Converts a Matplotlib cmap to pyqtgraphs colormaps. No dependency on matplotlib.
    Parameters:
    *cmap*: Cmap object. Imported from matplotlib.cm.*
    *nTicks*: Number of ticks to create when dict of functions is used. Otherwise unused.

    Copied from:
    https://github.com/honkomonk/pyqtgraph_sandbox/blob/master/mpl_cmaps_in_ImageView.py
    """

    # Case #1: a dictionary with 'red'/'green'/'blue' values as list of ranges (e.g. 'jet')
    # The parameter 'cmap' is a 'matplotlib.colors.LinearSegmentedColormap' instance ...
    if hasattr(cmap, '_segmentdata'):
        colordata = getattr(cmap, '_segmentdata')
        if ('red' in colordata) and isinstance(colordata['red'], collections.Sequence):

            # collect the color ranges from all channels into one dict to get unique indices
            posDict = {}
            for idx, channel in enumerate(('red', 'green', 'blue')):
                for colorRange in colordata[channel]:
                    posDict.setdefault(colorRange[0], [-1, -1, -1])[idx] = colorRange[2]

            indexList = list(posDict.keys())
            indexList.sort()
            # interpolate missing values (== -1)
            for channel in range(3):  # R,G,B
                startIdx = indexList[0]
                emptyIdx = []
                for curIdx in indexList:
                    if posDict[curIdx][channel] == -1:
                        emptyIdx.append(curIdx)
                    elif curIdx != indexList[0]:
                        for eIdx in emptyIdx:
                            rPos = (eIdx - startIdx) / (curIdx - startIdx)
                            vStart = posDict[startIdx][channel]
                            vRange = (posDict[curIdx][channel] - posDict[startIdx][channel])
                            posDict[eIdx][channel] = rPos * vRange + vStart
                        startIdx = curIdx
                        del emptyIdx[:]
            for channel in range(3):  # R,G,B
                for curIdx in indexList:
                    posDict[curIdx][channel] *= 255

            rgb_list = [[i, posDict[i]] for i in indexList]

        # Case #2: a dictionary with 'red'/'green'/'blue' values as functions (e.g. 'gnuplot')
        elif ('red' in colordata) and isinstance(colordata['red'], collections.Callable):
            indices = np.linspace(0., 1., nTicks)
            luts = [np.clip(np.array(colordata[rgb](indices), dtype=np.float), 0, 1) * 255 \
                    for rgb in ('red', 'green', 'blue')]
            rgb_list = zip(indices, list(zip(*luts)))

    # If the parameter 'cmap' is a 'matplotlib.colors.ListedColormap' instance, with the attributes 'colors' and 'N'
    elif hasattr(cmap, 'colors') and hasattr(cmap, 'N'):
        colordata = getattr(cmap, 'colors')
        # Case #3: a list with RGB values (e.g. 'seismic')
        if len(colordata[0]) == 3:
            indices = np.linspace(0., 1., len(colordata))
            scaledRgbTuples = [(rgbTuple[0] * 255, rgbTuple[1] * 255, rgbTuple[2] * 255) for rgbTuple in colordata]
            rgb_list = zip(indices, scaledRgbTuples)

        # Case #4: a list of tuples with positions and RGB-values (e.g. 'terrain')
        # -> this section is probably not needed anymore!?
        elif len(colordata[0]) == 2:
            rgb_list = [(idx, (vals[0] * 255, vals[1] * 255, vals[2] * 255)) for idx, vals in colordata]

    # Case #X: unknown format or datatype was the wrong object type
    else:
        raise ValueError("[cmapToColormap] Unknown cmap format or not a cmap!")

    # Convert the RGB float values to RGBA integer values
    return list([(pos, (int(r), int(g), int(b), 255)) for pos, (r, g, b) in rgb_list])


class FFTWorker(QObject):
    IN = pyqtSignal(object)
    OUT = pyqtSignal(object)

    def __init__(self, win, chunk_size, parent=None):
        super(FFTWorker, self).__init__(parent)
        self.win = win
        self.chunk_size = chunk_size
        self.IN.connect(self.psd)

    @pyqtSlot(object)
    def psd(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(chunk * self.win) / self.chunk_size
        psd = abs(spec) ** 2           # magnitude
        psd += 1e-5                    # Add fake noise floor
        psd = 20 * np.log10(psd)       # to dB scale
        self.OUT.emit(psd)


class SpectrogramWidget(pg.PlotWidget):
    """Live spectrogram widgets

    Based off code from here:
    http://amyboyle.ninja/Pyqtgraph-live-spectrogram
    """

    def __init__(
            self,
            config: PlotConfig,
            show_x=False,
        ):
        super(SpectrogramWidget, self).__init__()
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.showAxis("left", False)
        self.showAxis("bottom", show_x)
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array = None
        self._worker = None

        self.apply_config(config)

    def set_info(self, rate=None, chunk=None):
        if rate is not None:
            self._config["rate"] = rate
        if chunk is not None:
            self._config["chunk"] = chunk
        self.apply_config(self._config)

    def apply_config(self, config: PlotConfig):
        self._config = config
        self.chunk_size = config["chunk"]
        freq = (
            np.arange((self.chunk_size // 2) + 1) /
            (float(self.chunk_size) / config["rate"])
        )
        self.freq_mask = np.where(
            (config["spectrogram.min_freq"] < freq) &
            (freq < config["spectrogram.max_freq"])
        )[0]

        new_shape = (
            int(config["window"] * config["rate"] / config["chunk"]),
            len(freq[self.freq_mask])
        )
        if self.img_array is None or self.img_array.shape != new_shape:
            self.img_array = np.zeros(new_shape)

        self.cmap = self.get_cmap(config["spectrogram.cmap"])

        lut = self.cmap.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([config["spectrogram.min_level"], config["spectrogram.max_level"]])

        yscale = (
            1.0 /
            (self.img_array.shape[1] / freq[self.freq_mask][-1])
        )

        self.img.scale(1, yscale) # (1. / config["rate"]) * self.chunk_size, yscale)
        self.win = np.hanning(self.chunk_size)

        if self._worker:
            self._worker.win = self.win
            self._worker.chunk_size = self.chunk_size
        else:
            self._worker = FFTWorker(self.win, self.chunk_size)
            self._worker.OUT.connect(self.push_data)

        self.setXRange(0, self.img_array.shape[0])

    def get_cmap(self, cmap):
        if cmap is None:
            cmap = "afmhot"
        if isinstance(cmap, str):
            cmap = getattr(cm, cmap)

        pos = np.array([0.1, 0.7])
        color = np.array([
            (0, 0 , 0, 255),
            (255, 255, 255, 255)
        ], dtype=np.ubyte)

        pos, color = zip(*cmapToColormap(cmap))
        pos = 0.1 + 0.6 * np.array(pos)
        cmap = pg.ColorMap(pos, color)

        return cmap

    def receive_data(self, chunk):
        if self._worker:
            self._worker.IN.emit(chunk)

    @pyqtSlot(object)
    def push_data(self, psd):
        if self.img_array is not None:
            self.img_array = np.roll(self.img_array, -1, 0)
            self.img_array[-1:] = psd[self.freq_mask]

    def draw(self):
        if self.img_array is not None:
            self.img.setImage(self.img_array, autoLevels=False)


class WaveformWidget(pg.PlotWidget):
    """Live spectrogram widgets

    Based off code from here:
    http://amyboyle.ninja/Pyqtgraph-live-spectrogram
    """

    def __init__(
            self,
            config: PlotConfig,
            show_x=False,
        ):
        super(WaveformWidget, self).__init__()
        self.showAxis("left", False)
        self.showAxis("bottom", show_x)

        self.curve = None
        self._buffer = None
        self.threshold_line = None
        self.threshold = 0

        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.hideButtons()

        self.apply_config(config)

    def set_info(self, rate=None, chunk=None):
        if rate is not None:
            self._config["rate"] = rate
        if chunk is not None:
            self._config["chunk"] = chunk
        self.apply_config(self._config)

    def apply_config(self, config: PlotConfig):
        self._config = config

        if config["amplitude.show_max_only"]:
            target_maxlen = int(config["window"] * config["rate"] / config["chunk"])
        else:
            target_maxlen = int(config["window"] * config["rate"] / config["amplitude.downsample"])

        if self._buffer is None or self._buffer.maxlen != target_maxlen:
            self._buffer = collections.deque(maxlen=target_maxlen)
            self._buffer.extend(np.zeros(self._buffer.maxlen))

        if self.curve is None:
            self.curve = self.plot(np.array(self._buffer))
        else:
            self.curve.setData(np.array(self._buffer))

        if self.threshold_line is None:
            self.threshold_line = self.plot([0, self._buffer.maxlen], [self.threshold, self.threshold])
        else:
            self.set_threshold(self.threshold)

        if config["amplitude.show_max_only"]:
            ylim = (config["amplitude.y_min"], config["amplitude.y_max"])
        else:
            ylim = (-config["amplitude.y_max"], config["amplitude.y_max"])

        self.setYRange(*ylim, padding=0)
        self.setXRange(0, self._buffer.maxlen)

    def set_threshold(self, threshold):
        if self.threshold_line:
            self.threshold = threshold
            self.threshold_line.setData([0, self._buffer.maxlen], [threshold, threshold])

    def receive_data(self, chunk):
        # self._buffer.append(np.max(np.power(np.abs(chunk), 2)))
        # self._buffer.extend(np.power(np.abs(chunk), 2))
        # self._buffer.extend(chunk[::self.downsample])
        if self._buffer is not None:
            if self._config["amplitude.show_max_only"]:
                self._buffer.append(np.max(np.abs(chunk)))
            else:
                self._buffer.extend(chunk[::self._config["amplitude.downsample"]])

    def draw(self):
        if self.curve is not None:
            self.curve.setData(np.array(self._buffer).astype(np.float))
