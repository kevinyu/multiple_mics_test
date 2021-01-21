import numpy as np


def db_scale(x, dB=None):
    """Scale the channels of a signal (in dB) independently"""
    return (np.power(10.0, dB / 20.0) * x).astype(x.dtype)
