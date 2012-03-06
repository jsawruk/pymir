# test.py
# Generate test signals to verify algorithms
# For internal use only

import numpy
import scipy
import scipy.signal

# Generate a frequency sweep from 220Hz (A3) to 440Hz (A4).
def chirp():
    t = numpy.arange(0, 2 ,0.001)
    return scipy.signal.chirp(t, 220, 1.5, 440)