import ctypes
import platform


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def datetime2str(dt):
    s = (
        dt.hour * (1000 * 60 * 60) +
        dt.minute * (1000 * 60) +
        dt.second * 1000 +
        dt.microsecond / 1000
    )
    return "{}_{}".format(
        dt.strftime("%y%m%d"),
        int(s)
    )


def _set_thread_execution(state):
    ctypes.windll.kernel32.SetThreadExecutionState(state)


def prevent_standby():
    if platform.system() == 'Windows':
        _set_thread_execution(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def allow_standby():
    if platform.system() == 'Windows':
        _set_thread_execution(ES_CONTINUOUS)


def long_running(func):
    def inner(*args, **kwargs):
        prevent_standby()
        result = func(*args, **kwargs)
        allow_standby()
        return result
    return inner
