"""Testing color maps and windows

TODO: toggle channel views
TODO: zooming in and out in time
TODO: change redraw flag and time start info to be per file
"""
import curses
import curses.panel
import curses.textpad
import glob
import logging
import os
import time
from collections import namedtuple

import click
import soundfile
from soundsig.sound import spectrogram
import numpy as np
import scipy.ndimage
import cv2


SPECTROGRAM_LOWER_QUANTILE = 0.01
SPECTROGRAM_UPPER_QUANTILE = 0.99
DEFAULT_BG_COLOR = 232
MAX_TERMINAL_COLORS = 256
DEFAULT_BUFFER_CHARS = 1
SPECTROGRAM_SAMPLE_RATE = 500
SPECTROGRAM_FREQ_SPACING = 50
MIN_TIMESCALE = 0.01
MAX_TIMESCALE = 10.0
DEFAULT_TIME_STEP = 0.5


logger = logging.getLogger(__name__)
# fh = logging.FileHandler("specview.log")
# fh.setLevel(logging.DEBUG)
# logger.addHandler(fh)


PanelCoord = namedtuple("PanelCoord", [
    "nlines",
    "ncols",
    "y",
    "x"
])


class Colormap(object):

    def __init__(self, colors, bg_colors=None, bin_edges=None, bg_bin_edges=None):
        """A colormap defined by a set of colors

        Params
        ======
        colors : list
            List of integer values representing curses colors
        bg_colors : list (default=[-1])
            List of background colors to pair with foreground colors
            By default, uses a black background for all colors
        bin_edges : list (default=None)
            List (length of len(colors) - 1) of floats from (0, 1] representing
            partition of the space [0, 1], e.g. [0.33, 0.66] for 3 colors.
            By default, gives each color an equal partition.
        bg_bin_edges : list (default=None)
            List (length of len(colors) - 1) of floats from (0, 1] representing
            partition of the space [0, 1], e.g. [0.33, 0.66] for 3 colors
            By default, gives each color an equal partition.
        """
        self.colors = colors
        if bg_colors is None:
            self.bg_colors = self.default_bg_colors
        else:
            self.bg_colors = bg_colors

        if bin_edges is None:
            self.bin_edges = self.default_bin_edges(self.colors)
        else:
            self.bin_edges = bin_edges

        if bg_bin_edges is None:
            self.bg_bin_edges = self.default_bin_edges(self.bg_colors)
        else:
            self.bg_bin_edges = bg_bin_edges

        self.validate()

    def __iter__(self):
        for i in range(len(self.colors)):
            for j in range(len(self.bg_colors)):
                yield self.get_pair_idx(i, j)

    def default_bin_edges(self, colors):
        return np.arange(1, len(colors)) / len(colors)

    @property
    def default_bg_colors(self):
        # Black default background
        return [DEFAULT_BG_COLOR]

    def validate(self):
        if len(self.colors) * len(self.bg_colors) > MAX_TERMINAL_COLORS:
            raise ValueError("The number of combined fg x bg colors cannot exceed 256")

    def setup(self):
        """Map fg/bg color pairs

        Note that index 0 is fixed to white text on black background
        and cannot be changed. Thus the get_char function will provide
        an alternate character and color to account for this.

        TODO: we can get even more value from the 255 availble color slots
        by keeping in mind that all colors where the foreground and background
        colors match can be mapped to the same color with a █ charcter instead
        of a ▄. But it would make the code a bit more complicated.
        """
        for fg_idx, foreground in enumerate(self.colors):
            for bg_idx, background in enumerate(self.bg_colors):
                curses.init_pair(
                    self.get_pair_idx(fg_idx, bg_idx),
                    foreground,
                    background
                )

    def get_pair_idx(self, foreground_idx, background_idx=0):
        return (foreground_idx + len(self.colors) * background_idx)

    def _fg_frac(self, frac):
        return np.searchsorted(self.bin_edges, frac)

    def _bg_frac(self, frac):
        return np.searchsorted(self.bg_bin_edges, frac)

    def frac(self, frac, bg_frac=0):
        fg_idx = np.searchsorted(self.bin_edges, frac)
        bg_idx = np.searchsorted(self.bg_bin_edges, bg_frac)
        return self.get_pair_idx(fg_idx, bg_idx)

    def get_char_by_idx(self, idx):
        return "▄", curses.color_pair(idx)

    def get_char_by_frac(self, frac, bg_frac=0):
        """Return a character and color idx for the given foreground and bg fracs
        """
        color_idx = self.frac(frac, bg_frac)
        return "▄", curses.color_pair(color_idx)


class PairedColormap(Colormap):

    @property
    def default_bg_colors(self):
        return self.colors

    def setup(self):
        """Map fg/bg color pairs

        Note that index 0 is fixed to white text on black background
        and cannot be changed. Thus the get_char function will provide
        an alternate character and color to account for this.

        TODO: we can get even more value from the 255 availble color slots
        by keeping in mind that all colors where the foreground and background
        colors match can be mapped to the same color with a █ charcter instead
        of a ▄. But it would make the code a bit more complicated.
        """
        for fg_idx, foreground in enumerate(self.colors):
            for bg_idx, background in enumerate(self.bg_colors):
                if fg_idx == 0 and bg_idx == 0:
                    self._case00 = self.get_pair_idx(fg_idx, 1)
                else:
                    curses.init_pair(
                        self.get_pair_idx(fg_idx, bg_idx),
                        foreground,
                        background
                    )

    def get_pair_idx(self, foreground_idx, background_idx=0):
        return (foreground_idx + len(self.colors) * background_idx)

    def get_char_by_idx(self, idx):
        if idx == 0:
            return "█", curses.color_pair(self._case00)
        else:
            return "▄", curses.color_pair(idx)

    def get_char_by_frac(self, frac, bg_frac=0):
        """Return a character and color idx for the given foreground and bg fracs
        """
        color_idx = self.frac(frac, bg_frac)
        return self.get_char_by_idx(color_idx)


def get_colormaps():
    cmaps = {
        "full": Colormap(range(curses.COLORS)),
        "greys": PairedColormap([
            232, 233, 234, 235, 237,
            238, 239, 241, 242, 244,
            246, 248, 250, 252, 255
        ]),
        "plasma": PairedColormap([
            232, 17, 18, 57, 91,
            167, 205, 204, 203, 202,
            208, 214, 220, 227, 229
        ]),
        "viridis": PairedColormap([
            232, 17, 18, 20, 26,
            24, 22, 28, 34, 40,
            46, 112, 154, 190, 226
        ]),
        "blues": PairedColormap([
            232, 17, 18, 19, 20,
            21, 27, 33, 39, 45,
            51, 87, 123, 159, 195
        ])
    }
    for key in list(cmaps.keys()):
        if isinstance(cmaps[key], PairedColormap):
            cmaps["{}_r".format(key)] = PairedColormap(
            list(reversed(cmaps[key].colors))
        )
    return cmaps


def _view_colormap(stdscr, cmap=None):
    cmaps = get_colormaps()

    curses.use_default_colors()
    if cmap is None:
        cmap = cmaps.get("full")
    else:
        cmap = cmaps[cmap]

    cmap.setup()

    for color_idx in cmap:
        color_str = str(color_idx)
        full_str = (4 - len(color_str)) * " " + color_str
        row_idx = 1 + color_idx % (curses.LINES - 2)
        col_idx = (color_idx // (curses.LINES - 2)) * 5
        char, color = cmap.get_char_by_idx(color_idx)
        stdscr.addstr(row_idx, col_idx, full_str + char, color)

    while True:
        ch = stdscr.getch()
        if ch == ord("q"):
            break


def create_window(
        nlines,
        ncols,
        y,
        x,
        parent=None,
    ):
    """Create a window within a parent window
    """
    if parent is None:
        window = curses.newwin(nlines, ncols, y, x)
    else:
        window = parent.subwin(nlines, ncols, y, x)

    # Should this go outside?
    curses.panel.new_panel(window)
    curses.panel.update_panels()
    window.refresh()

    return window


def annotate_window(
        window,
        title=None,
        subtitle=None,
        page=None,
        border=None
    ):
    nlines, ncols = window.getmaxyx()

    title = title or getattr(window, "title", None)
    subtitle = subtitle or getattr(window, "subtitle", None)

    if border is not None:
        window.border(*border)
    if title:
        window.addstr(0, 1, title, curses.A_NORMAL)
    if subtitle:
        window.addstr(nlines - DEFAULT_BUFFER_CHARS, 1, subtitle, curses.A_NORMAL)

    if page is not None:
        page_string = str(page)
        window.addstr(
            nlines - DEFAULT_BUFFER_CHARS,
            ncols - DEFAULT_BUFFER_CHARS - len(page_string),
            page_string,
            curses.A_NORMAL)

    window.refresh()


def compute_layout(nlines, ncols, panel_lines=1, panel_cols=1, pady=(0, 0), padx=(0, 0)):
    """Compute where panels should live in coordinate system

    Params
    ======
    nlines : int
        number of rows in parent window
    ncols : int
        number of columns in parent window
    panel_lines :
        number of rows of panels to fit into outer window
    panel_cols :
        number of columns of panels to fit into parent window
    pady : tuple
        padding in rows on top and bottom (pad_top, pad_bottom)
    padx : tuple
        padding in columns on left and right (pad_left, pad_right)
    """
    y0 = pady[0]
    y1 = nlines - pady[1]
    x0 = padx[0]
    x1 = ncols - padx[1]

    parent_width = x1 - x0
    parent_height = y1 - y0

    panel_width = parent_width // panel_cols
    panel_height = parent_height // panel_lines

    coordinates = []
    for col in range(panel_cols):
        for row in range(panel_lines):
            coordinates.append(
                PanelCoord(
                    nlines=panel_height,
                    ncols=panel_width,
                    y=y0 + panel_height * row,
                    x=x0 + panel_width * col
                )
            )
    return coordinates


def _test_windows(stdscr, rows, cols):
    curses.use_default_colors()

    # Border main window
    stdscr.border(0)
    stdscr.addstr(0, 1, "Main Window", curses.A_BOLD)

    coords = compute_layout(
        curses.LINES,
        curses.COLS,
        panel_lines=rows,
        panel_cols=cols,
        padx=(DEFAULT_BUFFER_CHARS, DEFAULT_BUFFER_CHARS),
        pady=(DEFAULT_BUFFER_CHARS, DEFAULT_BUFFER_CHARS)
    )

    for i, coord in enumerate(coords):
        window = create_window(*coord, parent=stdscr, title="Window {}".format(i), page=i)

    stdscr.addstr(curses.LINES - 1, 1, "Press q to close:")

    cont = True
    while cont:
        ch = stdscr.getch()
        if ch == ord("q"):
            cont = False
            stdscr.clear()
        elif ch == ord("l"):
            pass


def draw(window, arr, y0, x0, cmap):
    """
    Draw data in arr into window, doubling y resolution using ▀ chars
    """
    floor = np.quantile(arr, SPECTROGRAM_LOWER_QUANTILE)
    ceil = np.quantile(arr, SPECTROGRAM_UPPER_QUANTILE)

    y1 = y0 + (arr.shape[0] // 2)  # Each char represents 2 freq bins
    x1 = x0 + arr.shape[1]

    for output_row in range(arr.shape[0] // 2):
        for output_col in range(arr.shape[1]):
            data = arr[output_row * 2:(output_row + 1) * 2, output_col]

            frac = (data[0] - floor) / (ceil - floor)
            bg_frac = (data[1] - floor) / (ceil - floor)
            char, color = cmap.get_char_by_frac(frac, bg_frac)

            window.addstr(
                y1 - DEFAULT_BUFFER_CHARS - output_row,
                x0 + DEFAULT_BUFFER_CHARS + output_col,
                char,
                color
            )


def get_soundfile_metadata(filename):
    with soundfile.SoundFile(filename) as f:
        return {
            "sampling_rate": f.samplerate,
            "frames": f.frames,
            "n_channels": f.channels,
        }


def wav_to_curses_spectrogram(filename, window, cmap, t_start=None, duration=None):
    """Draw spectrogram in a curses window

    Returns time, freq, and spectrogram arrays, as well as the duration of the audio file
    """
    lines, cols = window.getmaxyx()

    metadata = get_soundfile_metadata(filename)

    full_duration = metadata["frames"] / metadata["sampling_rate"]
    if duration:
        show_duration = duration
    else:
        show_duration = full_duration

    if t_start:
        read_idx = int(np.round(t_start * metadata["sampling_rate"]))
    else:
        read_idx = 0

    if show_duration:
        read_samples = int(np.floor(show_duration * metadata["sampling_rate"]))
    else:
        read_samples = metadata["frames"]

    # If we are too close to the end, show the last bit
    if read_idx + read_samples > metadata["frames"]:
        read_idx = max(0, metadata["frames"] - read_samples)
        read_samples = metadata["frames"] - read_idx

    # Read spectrogram
    logger.warning("DEBUG: {} {}".format(read_samples, read_idx))
    data, samplerate = soundfile.read(filename, read_samples, read_idx)

    # if data.ndim > 1:
    #     data = data[0]

    # Compute spec of shape (freqs, time) or (y, x)
    t, f, spec, _ = spectrogram(
        data,
        samplerate,
        SPECTROGRAM_SAMPLE_RATE,
        SPECTROGRAM_FREQ_SPACING,
        min_freq=1000,
        max_freq=8000,
        cmplx=False
    )

    # Resample spec to window size (with 1 padding and x2 resolution on y axis)
    y0 = DEFAULT_BUFFER_CHARS
    y1 = lines - DEFAULT_BUFFER_CHARS - y0
    x0 = DEFAULT_BUFFER_CHARS
    x1 = cols - DEFAULT_BUFFER_CHARS - x0

    spec_width = x1 - x0
    spec_height = (y1 - y0) * 2

    # OpenCV flips the dimensions
    resized_t = np.linspace(t[0], t[-1], spec_width)
    resized_f = np.linspace(f[0], f[-1], spec_height)
    resized_spec = cv2.resize(
        spec,
        dsize=(spec_width, spec_height),
        interpolation=cv2.INTER_CUBIC
    )
    draw(window, resized_spec, y0, x0, cmap)
    window.refresh()

    return resized_t, resized_f, resized_spec, show_duration, full_duration


class ViewState:
    def __init__(
            self,
            rows,
            cols,
            n_files,
            current_selection_idx,
            page_idx,
            cmap,
            time_start=None,
            time_scale=None,
        ):
        self.rows = rows
        self.cols = cols
        self.n_files = n_files
        self.current_selection_idx = current_selection_idx
        self.page_idx = page_idx
        self.time_scale = None
        self.time_start = 0.0
        self.windows = []
        self.window_annotations = []
        self.cmap = cmap

        self._needs_redraw = True

    def needs_redraw(self):
        return self._needs_redraw

    def mark_for_redraw(self):
        self._needs_redraw = True
        self.windows = []
        self.window_annotations = []

    def mark_redrawn(self):
        self._needs_redraw = False

    def update_colormap(self, cmap):
        self.cmap = cmap
        self.cmap.setup()

    @property
    def visible(self):
        return slice(self.page_idx, self.page_idx + self.n_visible)

    @property
    def n_visible(self):
        return self.rows * self.cols

    def page_down(self):
        new_page_idx = max(0, self.page_idx - self.rows)
        if self.page_idx != new_page_idx:
            self.mark_for_redraw()
            self.page_idx = new_page_idx

    def page_up(self):
        new_page_idx = min(self.max_page, self.page_idx + self.rows)
        if self.page_idx != new_page_idx:
            self.mark_for_redraw()
            self.page_idx = new_page_idx

    @property
    def max_page(self):
        n_files_rounded = int(np.ceil(self.n_files / self.n_visible)) * self.n_visible
        return n_files_rounded - self.n_visible

    @property
    def time_step(self):
        if self.time_scale:
            time_step = self.time_scale / 4
        else:
            time_step = DEFAULT_TIME_STEP
        return time_step

    def time_left(self):
        self.time_start = max(0, self.time_start - self.time_step)
        self.mark_for_redraw()

    def time_right(self):
        self.time_start = self.time_start + self.time_step
        self.mark_for_redraw()

    def left(self):
        self.current_selection_idx = max(0, self.current_selection_idx - self.rows)
        if self.current_selection_idx < self.page_idx:
            self.page_down()

    def right(self):
        self.current_selection_idx = min(self.n_files - 1, self.current_selection_idx + self.rows)
        if self.current_selection_idx > self.page_idx + self.n_visible - 1:
            self.page_up()

    def up(self):
        self.current_selection_idx = max(0, self.current_selection_idx - 1)
        if self.current_selection_idx < self.page_idx:
            self.page_down()

    def down(self):
        self.current_selection_idx = min(self.n_files - 1, self.current_selection_idx + 1)
        if self.current_selection_idx > self.page_idx + self.n_visible - 1:
            self.page_up()


def prompt(stdscr, msg):
    resp = ""

    stdscr.addstr(
        stdscr.getmaxyx()[0] - DEFAULT_BUFFER_CHARS,
        DEFAULT_BUFFER_CHARS,
        msg + (stdscr.getmaxyx()[1] - 2 * DEFAULT_BUFFER_CHARS - len(msg)) * " "
    )
    stdscr.refresh()

    resp_window = stdscr.subwin(
        1,
        stdscr.getmaxyx()[1] - 2 * DEFAULT_BUFFER_CHARS - len(msg),
        stdscr.getmaxyx()[0] - DEFAULT_BUFFER_CHARS,
        DEFAULT_BUFFER_CHARS + len(msg),
    )
    resp_input = curses.textpad.Textbox(resp_window)
    resp_input.edit()

    resp = resp_input.gather()
    del resp_window

    return str(resp)


def draw_instructions(stdscr):
    maxx = stdscr.getmaxyx()[1] - 2 * DEFAULT_BUFFER_CHARS

    instructions = [
        "[q] quit",
        "[←|↓|↑|→] select",
        "[r] rows",
        "[c] columns",
        "[t] jump to time"
        "[s] set time scale",
        "[shift + ←|↓|↑|→] scroll time",
        "[m] colormap",
        "[+|-] zoom",
    ]

    text = ""

    if maxx < len(text):
        # Don't error if the screen is too small for some reason...
        return

    for instruction in instructions:
        if len(text) + len(instruction) + 1 > maxx:
            break
        else:
            text += " " + instruction

    stdscr.addstr(
        stdscr.getmaxyx()[0] - DEFAULT_BUFFER_CHARS,
        DEFAULT_BUFFER_CHARS,
        text
    )


def _view_wavs(stdscr, rows, cols, files, cmap="greys"):
    """View wav files spectrograms in multiple windows"""
    cmaps = get_colormaps()
    curses.use_default_colors()

    # Set up colormap
    if cmap is None:
        cmap = cmaps.get("full")
    else:
        cmap = cmaps[cmap]
    cmap.setup()

    view_state = ViewState(
        rows,
        cols,
        len(files),
        0,
        0,
        cmap,
    )

    def redraw(view_state):
        if view_state.needs_redraw():
            stdscr.erase()
            stdscr.refresh()

        # Check that the number of rows and columns is allowable by the window size
        max_rows = (curses.LINES - 2 * DEFAULT_BUFFER_CHARS) // 9
        max_cols = (curses.COLS - 2 * DEFAULT_BUFFER_CHARS) // 9

        view_state.rows = min(max_rows, view_state.rows)
        view_state.cols = min(max_cols, view_state.cols)

        # Border main window
        # stdscr.border(0)
        # stdscr.addstr(0, 1, "Spectrograms", curses.A_BOLD)

        coords = compute_layout(
            curses.LINES,
            curses.COLS,
            panel_lines=view_state.rows,
            panel_cols=view_state.cols,
            padx=(DEFAULT_BUFFER_CHARS, DEFAULT_BUFFER_CHARS),
            pady=(DEFAULT_BUFFER_CHARS, DEFAULT_BUFFER_CHARS)
        )

        if view_state.needs_redraw():
            view_state.windows = []

        for i, (filename, coord) in enumerate(zip(
                files[view_state.visible], coords)):

            if view_state.needs_redraw():
                window = create_window(*coord)
                _, _, _, show_duration, full_duration = wav_to_curses_spectrogram(
                    filename,
                    window,
                    view_state.cmap,
                    view_state.time_start,
                    view_state.time_scale
                )
                view_state.windows.append(window)
                if show_duration == full_duration:
                    duration_string = "{:.2f}s".format(show_duration)
                else:
                    duration_string = "{:.2f}s/{:.2f}s".format(show_duration, full_duration)

                view_state.window_annotations.append({
                    "title": os.path.basename(filename),
                    "subtitle": duration_string,
                    "page": view_state.page_idx + i
                })
                view_state.window_states.append({
                    "full_duration": full_duration,
                    "time_start": 0.0
                })
            else:
                window = view_state.windows[i]

            if view_state.page_idx + i == view_state.current_selection_idx:
                annotate_window(
                    window,
                    border=(1, 1, 1, 1),
                    **view_state.window_annotations[i]
                )
            else:
                annotate_window(
                    window,
                    border=(0,),
                    **view_state.window_annotations[i]
                )
            window.refresh()

        page_string = str("*{}/{}".format(
            view_state.current_selection_idx,
            view_state.n_files - 1
        ))

        stdscr.addstr(
            stdscr.getmaxyx()[0] - DEFAULT_BUFFER_CHARS,
            stdscr.getmaxyx()[1] - DEFAULT_BUFFER_CHARS - len(page_string),
            page_string,
            curses.A_NORMAL)

        draw_instructions(stdscr)

        view_state.mark_redrawn()

    cont = True
    last_size = stdscr.getmaxyx()
    current_selection = 0

    redraw(view_state)

    while cont:
        ch = stdscr.getch()
        if ch == ord("q"):
            cont = False
            stdscr.clear()
        elif ch == curses.KEY_RESIZE:
            curr_size = stdscr.getmaxyx()
            if curr_size != last_size:
                view_state.mark_for_redraw()
                curses.resizeterm(*curr_size)
                stdscr.clear()
                stdscr.refresh()
                redraw(view_state)
                last_size = curr_size
        elif ch == curses.KEY_SLEFT or ch == ord("H"):
            view_state.time_left()
            redraw(view_state)
        elif ch == curses.KEY_SRIGHT or ch == ord("L"):
            view_state.time_right()
            redraw(view_state)
        elif ch == curses.KEY_LEFT or ch == ord("h"):
            view_state.left()
            redraw(view_state)
        elif ch == curses.KEY_RIGHT or ch == ord("l"):
            view_state.right()
            redraw(view_state)
        elif ch == curses.KEY_UP or ch == ord("k"):
            view_state.up()
            redraw(view_state)
        elif ch == curses.KEY_DOWN or ch == ord("j"):
            view_state.down()
            redraw(view_state)
        elif ch == curses.KEY_DOWN or ch == ord("s"):
            resp = prompt(stdscr, "Set timescale (max {}): ".format(MAX_TIMESCALE))
            if resp is None or not resp.strip():
                view_state.time_scale = None
            else:
                try:
                    scale = float(resp)
                except ValueError:
                    pass
                else:
                    if MIN_TIMESCALE < scale <= MAX_TIMESCALE:
                        view_state.time_scale = scale
            view_state.mark_for_redraw()
            redraw(view_state)
        elif ch == ord("r"):
            resp = prompt(stdscr, "Set rows [0-9]: ")
            if resp is not None:
                try:
                    rows = int(resp)
                except ValueError:
                    pass
                else:
                    if 0 <= rows <= 9:
                        view_state.rows = rows
            view_state.mark_for_redraw()
            redraw(view_state)
        elif ch == ord("c"):
            resp = prompt(stdscr, "Set cols [0-9]: ")
            if resp is not None:
                try:
                    cols = int(resp)
                except ValueError:
                    pass
                else:
                    if 0 <= cols <= 9:
                        view_state.cols = cols
            view_state.mark_for_redraw()
            redraw(view_state)
        elif ch == ord("m"):
            resp = prompt(stdscr, "Choose colormap ['greys', 'viridis', 'plasma', 'blues']: ")
            resp = resp.strip()
            if resp in cmaps.keys():
                view_state.update_colormap(cmaps[resp])
            view_state.mark_for_redraw()
            redraw(view_state)


@click.group()
def cli():
    pass


@click.command()
@click.argument("filenames", nargs=-1, type=click.Path(exists=True))
@click.option("-r", "--rows", type=int, default=1)
@click.option("-c", "--cols", type=int, default=1)
@click.option("--cmap", type=str, default="greys")
def load_wavs(filenames, rows, cols, cmap):
    panels = rows * cols
    files = []
    for filename in filenames:
        if not os.path.isdir(filename):
            files.append(filename)
        else:
            for _filename in glob.glob(os.path.join(filename, "*.wav")):
                files.append(_filename)

    curses.wrapper(_view_wavs, rows, cols, files, cmap=cmap)


@click.command()
@click.option("--cmap", type=str, default=None)
def view_colormap(cmap):
    curses.wrapper(_view_colormap, cmap)


@click.command()
@click.option("-r", "--rows", type=int, default=1)
@click.option("-c", "--cols", type=int, default=1)
def test_windows(rows, cols):
    curses.wrapper(_test_windows, rows, cols)


cli.add_command(view_colormap)
cli.add_command(test_windows)
cli.add_command(load_wavs)


if __name__ == "__main__":
    cli()
