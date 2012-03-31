# energy.py
# Compute the energy of the signal

import numpy
import scipy
from numpy.lib import stride_tricks
from pymir.audio.transform import *

def energy(signal):
    """
    This function does something.

    Example:
    >>> from test import chirp
    >>> s = chirp()
    >>> e = energy(s)
    >>> e
    array([ 0.26917694,  0.26901879,  0.26918094, ...,  0.18757919,
            0.18656895,  0.18561012])
    """
    N = len(signal)

    windowSize = 256
    window = numpy.hamming(windowSize)
    window.shape = (256,1)
    
    n = N - windowSize #number of windowed samples.

    # Create a view of signal who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    p = numpy.power(signal,2)
    s = stride_tricks.as_strided(p,shape=(n,windowSize), strides=(signal.itemsize,signal.itemsize))
    e = numpy.dot(s,window) / windowSize
    e.shape = (e.shape[0],)
    return e

def _test():
    import doctest
    doctest.testmod(verbose=True)
if __name__ == '__main__':
    _test()
