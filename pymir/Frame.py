# Frame class
# ndarray subclass for time-series data

import math

import numpy
import numpy.fft
from numpy import *

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