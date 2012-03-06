# energy.py
# Compute the energy of the signal

import numpy
import scipy

def energy(signal):
    N = len(signal)

    windowSize = 256
    window = numpy.hamming(windowSize)
    
    e = scipy.zeros(N)
    
    for i in range(0, N - windowSize):
        e[i] = numpy.sum(window * numpy.power(signal[i:i + windowSize], 2) ) / windowSize
    
            
    return e