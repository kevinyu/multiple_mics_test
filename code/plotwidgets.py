import collections
import matplotlib.cm as cm

import pyqtgraph as pg
import numpy as np

from PyQt5.QtCore import pyqtSignal, QObject, QThread, pyqtSlot

from settings import Settings


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


class FFTThread(QObject):
    IN = pyqtSignal(object)
    OUT = pyqtSignal(object)

    def __init__(self, win, chunk_size, parent=None):
        self.win = win
        self.chunk_size = chunk_size
        super(FFTThread, self).__init__(parent)

    @pyqtSlot(object)
    def psd(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(chunk * self.win) / self.chunk_size
        psd = abs(spec) ** 2           # magnitude
        psd += 1e-5                    # Add fake noise floor
        psd = 20 * np.log10(psd)       # to dB scale
        self.OUT.emit(psd)

    def run(self):
        self._thread = QThread(self)
        self.moveToThread(self._thread)
        self._thread.start()
        self.IN.connect(self.psd)


class SpectrogramWidget(pg.PlotWidget):
    """Live spectrogram widgets

    Based off code from here:
    http://amyboyle.ninja/Pyqtgraph-live-spectrogram
    """

    def __init__(
            self,
            chunk_size,
            min_freq=500,
            max_freq=8000,
            show_x=False,
            window=5,
            cmap=None
        ):

        super(SpectrogramWidget, self).__init__()
        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.showAxis("left", False)
        self.showAxis("bottom", show_x)
        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.chunk_size = chunk_size
        freq = (
            np.arange((self.chunk_size // 2) + 1) /
            (float(self.chunk_size) / Settings.RATE)
        )
        self.freq_mask = np.where(
            (min_freq < freq) &
            (freq < max_freq)
        )[0]

        self.img_array = np.zeros((
            int(window * Settings.RATE / Settings.CHUNK),
            len(freq[self.freq_mask])
        ))
        self.cmap = self.get_cmap() if cmap is None else cmap

        lut = self.cmap.getLookupTable(0.0, 1.0, 256)

        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        yscale = (
            1.0 /
            (self.img_array.shape[1] / freq[self.freq_mask][-1])
        )

        self.img.scale((1. / Settings.RATE) * self.chunk_size, yscale)
        self.win = np.hanning(self.chunk_size)

        self._worker = FFTThread(self.win, self.chunk_size)
        self._worker.run()

        self._worker.OUT.connect(self.push_data)

    def get_cmap(self):
        pos = np.array([0.1, 0.7])
        color = np.array([
            (0, 0 , 0, 255),
            (255, 255, 255, 255)
        ], dtype=np.ubyte)

        pos, color = zip(*cmapToColormap(cm.afmhot))
        pos = 0.1 + 0.6 * np.array(pos)
        cmap = pg.ColorMap(pos, color)

        return cmap

    def receive_data(self, chunk):
        self._worker.IN.emit(chunk)

    @pyqtSlot(object)
    def push_data(self, psd):
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = psd[self.freq_mask]

    def draw(self):
        self.img.setImage(self.img_array, autoLevels=False)


class WaveformWidget(pg.PlotWidget):
    """Live spectrogram widgets

    Based off code from here:
    http://amyboyle.ninja/Pyqtgraph-live-spectrogram
    """

    def __init__(
            self,
            chunk_size,
            show_x=False,
            window=5,
            ylim=(0, Settings.MAX_POWER_THRESHOLD),
            threshold=Settings.DEFAULT_POWER_THRESHOLD,
        ):
        super(WaveformWidget, self).__init__()
        self._buffer = collections.deque(
            maxlen=int(window * Settings.RATE / Settings.CHUNK)
        )
        self._buffer.extend(np.zeros(self._buffer.maxlen))

        self.showAxis("left", False)
        self.showAxis("bottom", show_x)

        self.chunk_size = chunk_size
        self.curve = self.plot(np.array(self._buffer))
        self.threshold = threshold
        self.threshold_line = self.plot([0, self._buffer.maxlen], [self.threshold, self.threshold])

        self.setMouseEnabled(False, False)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setYRange(*ylim, padding=0)

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.threshold_line.setData([0, self._buffer.maxlen], [threshold, threshold])

    def receive_data(self, chunk):
        # self._buffer.append(np.max(np.power(np.abs(chunk), 2)))
        # self._buffer.extend(np.power(np.abs(chunk), 2))
        self._buffer.append(np.max(np.abs(chunk)))

    def draw(self):
        self.curve.setData(np.array(self._buffer))

