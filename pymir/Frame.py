# Frame class
# ndarray subclass for time-series data

import math
from math import *

import numpy
import numpy.fft
from numpy import *

from numpy.lib import stride_tricks

import scipy

from pymir import Spectrum

class Frame(numpy.ndarray):
    
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
          strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = numpy.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)
        
        obj.sampleRate = 0
        
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        
        self.sampleRate = getattr(obj, 'sampleRate', None)
        
        # We do not need to return anything
        
    #####################
    # Frame methods
    #####################
    
    # Constant Q Transform (CQT)
    def cqt(self):
        N = len(self)
        y = array(zeros(N))
        a = sqrt(2 / float(N))
        for k in range(N):
            for n in range(N):
                y[k] += self[n] * cos(pi * (2*n + 1) * k / float(2 * N))
            if k == 0:
                y[k] = y[k] * sqrt(1 / float(N))
            else:
                y[k] = y[k] * a
        return y
    
    # Discrete Cosine Transform (DCT)
    def dct(self):
        N = len(self)
        y = array(zeros(N))
        a = sqrt(2 / float(N))
        for k in range(N):
            for n in range(N):
                y[k] += self[n] * cos(pi * (2*n + 1) * k / float(2 * N))
            if k == 0:
                y[k] = y[k] * sqrt(1/float(N))
            else:
                y[k] = y[k] * a
        return y
    
    # Energy
    def energy(self):
        N = len(self)

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
    
    # Root-mean-squared amplitude
    def rms(self):
        sum = 0
        for i in range(0, len(self)):
            sum = sum + self[i] ** 2
            
        sum = sum / (1.0 * len(self))
        
        return math.sqrt(sum)
    
    # Spectrum
    def spectrum(self):
        fftdata = numpy.fft.rfft(self)
        spectrum = fftdata.view(Spectrum.Spectrum)
        spectrum.sampleRate = self.sampleRate
        
        return spectrum
    
    # Zero-crossing rate (ZCR)
    def zcr(self):
        zcr = 0
        for i in range(1, len(self)):
            if (self[i - 1] * self[i]) < 0:
                zcr = zcr + 1
                
        return zcr / (1.0 * len(self))