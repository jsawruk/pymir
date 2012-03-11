# energy.py
# Compute the energy of the signal

import numpy
import scipy
from numpy.lib import stride_tricks

def energy(signal):
    """
    This function does something.

    Example:
    >>> from test import chirp
    >>> s = chirp()
    >>> e = energy(s)

    Below is for testing that I don't break things
    >>> import hashlib
    >>> hashlib.md5(e).hexdigest()
    '615c65bb9e9055a75c32ceb5d309e597'
    >>> hashlib.sha1(e).hexdigest()
    '531b9ba53419bec6d4f679adc918e7449e17e452'
    """
    N = len(signal)

    windowSize = 256
    window = numpy.hamming(windowSize)
    
    n = N - windowSize #number of windowed samples.

    # Create a view of signal who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    s = stride_tricks.as_strided(signal,shape=(n,windowSize), strides=(signal.itemsize,signal.itemsize))
    e = (window * numpy.power(s, 2)).sum(1) / windowSize
    return e

def _test():
    import doctest
    doctest.testmod(verbose=True)
if __name__ == '__main__':
    _test()
