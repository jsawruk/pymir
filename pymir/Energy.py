""" 
energy.py
Compute energy and related quantities
Last updated: 15 December 2012
"""

import numpy
import scipy
from numpy.lib import stride_tricks

def energy(audioData, windowSize = 256):
    """
    Compute the energy of the given audio data, using the given windowSize

    Example:
    >>> from test import chirp
    >>> s = chirp()
    >>> e = energy(s)
    >>> e
    array([ 0.26917694,  0.26901879,  0.26918094, ...,  0.18757919,
            0.18656895,  0.18561012])
    """
    N = len(audioData)

    window = numpy.hamming(windowSize)
    window.shape = (windowSize, 1)
    
    n = N - windowSize # number of windowed samples.

    # Create a view of audioData who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    p = numpy.power(audioData, 2)
    s = stride_tricks.as_strided(p, shape=(n, windowSize), strides=(audioData.itemsize, audioData.itemsize))
    e = numpy.dot(s, window) / windowSize
    e.shape = (e.shape[0], )
    return e

def dEnergy(audioData, windowSize = 256):
    """
    Compute the dEnergy differential term with windowing
    """
    e = energy(audioData, windowSize)
    diffE = numpy.diff(e)

    N = len(diffE)

    window = numpy.hamming(windowSize)
    window.shape = (windowSize, 1)
    
    n = N - windowSize # number of windowed samples.

    # Create a view of diffE who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    p = numpy.power(diffE, 2)
    s = stride_tricks.as_strided(p, shape=(n, windowSize), strides=(diffE.itemsize, diffE.itemsize))
    dE = numpy.dot(s, window) / windowSize
    dE.shape = (dE.shape[0], )
    return dE

def dLogEnergy(audioData, windowSize = 256):
    """
    Compute d(log(Energy)) with windowing
    """
    e = energy(audioData, windowSize)
    logE = numpy.log(e)
    diffLogE = numpy.diff(logE)

    N = len(diffLogE)

    window = numpy.hamming(windowSize)
    window.shape = (windowSize, 1)
    
    n = N - windowSize # number of windowed samples.

    # Create a view of diffLogE who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
    p = numpy.power(diffLogE, 2)
    s = stride_tricks.as_strided(p, shape=(n, windowSize), strides=(diffLogE.itemsize, diffLogE.itemsize))
    dLogE = numpy.dot(s, window) / windowSize
    dLogE.shape = (dLogE.shape[0], )
    return dLogE

def _test():
    import doctest
    doctest.testmod(verbose=True)
if __name__ == '__main__':
    _test()
