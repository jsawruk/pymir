"""
Spectrum class
ndarray subclass for spectral data
Last updated: 9 December 2012
"""
from __future__ import division

import math

import numpy
from numpy import *

from numpy import fft,array,arange,zeros,dot,transpose
from math import sqrt,cos,pi

import pymir

class Spectrum(numpy.ndarray):
    
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
    # Spectrum methods
    #####################
    
    def centroid(self):
        """
        Compute the spectral centroid
        """
        binNumber = 0
        
        numerator = 0
        denominator = 0
        
        for bin in self:
            # Compute center frequency
            f = (self.sampleRate / 2.0) / len(self) 
            f = f * binNumber
            
            numerator = numerator + (f * abs(bin))
            denominator = denominator + abs(bin)
            
            binNumber = binNumber + 1
            
        return (numerator * 1.0) / denominator
    
    def chroma(self):
        """
        Compute the 12-ET chroma vector from this spectrum
        """
        chroma = [0] * 12
        for index in range(0, len(self)):
            
            # Assign a frequency value to each bin
            f = index * (self.sampleRate / 2.0) / len(self)
            
            # Convert frequency to pitch to pitch class
            if f != 0:
                pitch = int(round(69 + 12 * math.log(f / 440.0, 2)))
            else:
                pitch = 0
            pitchClass = pitch % 12
            
            chroma[pitchClass] = chroma[pitchClass] + abs(self[index])
        
        # Normalize the chroma vector
        maxElement = max(chroma)
        chroma = [c / maxElement for c in chroma]
        
        return chroma
    
    def idct(self):
        """
        Compute the Inverse Discrete Cosine Transform (IDCT)
        """
        N = len(self)
        x = array(zeros(N))
        a = sqrt(2 / float(N))
        for n in range(N):
            for k in range(N):
                if k == 0:
                    x[n] += sqrt(1 / float(N)) * self[k] * cos(pi * (2*n + 1) * k / float(2 * N))
                else:
                    x[n] += a * self[k] * cos(pi * (2*n + 1) * k / float(2 * N))
        return x
    
    def ifft(self):
        """
        Compute the Inverse FFT
        """
        fftdata = numpy.fft.irfft(self)
        frame = fftdata.view(pymir.Frame)
        frame.sampleRate = self.sampleRate
        
        return frame

    def mfcc(self, m, NumFilters = 48):
        """
        Compute the Mth Mel-Frequency Cepstral Coefficient
        """
        result = 0
        outerSum = 0
        innerSum = 0

        binSize = len(self)

        if m >= NumFilters:
            return 0    # This represents an error condition - the specified coefficient is greater than or equal to the number of filters. The behavior in this case is undefined.

        result = self.mfccNormalizationFactor(NumFilters, m)

        for filterBand in range(1, NumFilters + 1):
            # Compute inner sum
            innerSum = 0
            for frequencyBand in range(0, binSize - 1):
                innerSum = innerSum + abs(self[frequencyBand] * self.mfccFilterParameter(binSize, frequencyBand, filterBand))

            if innerSum > 0:
                innerSum = log(innerSum) # The log of 0 is undefined, so don't use it

            innerSum = innerSum * math.cos(((m * math.pi) / NumFilters) * (filterBand - 0.5))
            
            outerSum = outerSum + innerSum

        result = result * outerSum

        return result

    def mfccNormalizationFactor(self, NumFilters, m):
        """
        Intermediate computation used by mfcc function. 
        Computes a normalization factor
        """
        normalizationFactor = 0

        if m == 0:
            normalizationFactor = math.sqrt(1.0 / NumFilters)
        else:
            normalizationFactor = math.sqrt(2.0 / NumFilters)

        return normalizationFactor

    def mfccFilterParameter(self, binSize, frequencyBand, filterBand):
        """
        Intermediate computation used by the mfcc function. 
        Compute the filter parameter for the specified frequency and filter bands
        """
        filterParameter = 0
        boundary = (frequencyBand * self.samplingRate) / float(binSize)     # k * Fs / N
        prevCenterFrequency = self.mfccGetCenterFrequency(filterBand - 1)   # fc(l - 1)
        thisCenterFrequency = self.mfccGetCenterFrequency(filterBand)       # fc(l)
        nextCenterFrequency = self.mfccGetCenterFrequency(filterBand + 1)   # fc(l + 1)

        if boundary >= 0 and boundary < prevCenterFrequency:
            filterParameter = 0
        
        elif boundary >= prevCenterFrequency and boundary < thisCenterFrequency:
            filterParameter = (boundary - prevCenterFrequency) / (thisCenterFrequency - prevCenterFrequency)
            filterParameter = filterParameter * self.mfccGetMagnitudeFactor(filterBand)
        
        elif boundary >= thisCenterFrequency and boundary < nextCenterFrequency:
                filterParameter = (boundary - nextCenterFrequency) / (thisCenterFrequency - nextCenterFrequency)
                filterParameter = filterParameter * self.mfccGetMagnitudeFactor(filterBand)
        
        elif boundary >= nextCenterFrequency and boundary < samplingRate:
                filterParameter = 0

        return filterParameter

    def mfccGetMagnitudeFactor(self, filterBand):
        """
        Intermediate computation used by the mfcc function. 
        Compute the band-dependent magnitude factor for the given filter band
        """
        magnitudeFactor = 0
        
        if filterBand >= 1 and filterBand <= 14:
            magnitudeFactor = 0.015
        elif filterBand >= 15 and filterBand <= 48:
            magnitudeFactor = 2.0 / (self.mfccGetCenterFrequency(filterBand + 1) - self.mfccGetCenterFrequency(filterBand - 1))

        return magnitudeFactor

    def mfccGetCenterFrequency(self, filterBand):
        """
        Intermediate computation used by the mfcc function. 
        Compute the center frequency (fc) of the specified filter band (l)
        This where the mel-frequency scaling occurs. Filters are specified so that their
        center frequencies are equally spaced on the mel scale
        """
        centerFrequency = 0

        if filterBand == 0:
            centerFrequency = 0;
        elif filterBand >= 1 and filterBand <= 14:
            centerFrequency = (200.0 * filterBand) / 3.0
        else:
            exponent = filterBand - 14
            centerFrequency = math.pow(1.0711703, exponent)
            centerFrequency = centerFrequency * 1073.4
        
        return centerFrequency

    # TODO
    # Bandwidth    
    # Cepstrum?
    # Crest
    # Flatness
    # Kurtosis
    # MFCCs?
    # Rolloff
    # Skewness
    # Spread
    # Tilt
    