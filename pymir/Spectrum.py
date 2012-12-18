"""
Spectrum class
ndarray subclass for spectral data
Last updated: 17 December 2012
"""
from __future__ import division

import math

import numpy
from numpy import *

import scipy.stats
import scipy.stats.mstats

import matplotlib.pyplot as plt

import pymir
from pymir import MFCC, Pitch, Transforms

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
        Compute the spectral centroid.
        Characterizes the "center of gravity" of the spectrum.
        Approximately related to timbral "brightness"
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
        return Pitch.chroma(self)

    def crest(self):
        """
        Compute the spectral crest factor, i.e. the ratio of the maximum of the spectrum to the 
        sum of the spectrum
        """
        absSpectrum = abs(self)
        spectralSum = numpy.sum(absSpectrum)

        maxFrequencyIndex = numpy.argmax(absSpectrum)
        maxSpectrum = absSpectrum[maxFrequencyIndex]

        return maxSpectrum / spectralSum

    def flatness(self):
        """
        Compute the spectral flatness (ratio between geometric and arithmetic means)
        """
        geometricMean = scipy.stats.mstats.gmean(abs(self))
        arithmeticMean = self.mean()

        return geometricMean / arithmeticMean
        
    def idct(self):
        """
        Compute the Inverse Discrete Cosine Transform (IDCT)
        """
        return Transforms.idct(self)
    
    def ifft(self):
        """
        Compute the Inverse FFT
        """
        return Transforms.ifft(self)

    def kurtosis(self):
        """
        Compute the spectral kurtosis (fourth spectral moment)
        """
        return scipy.stats.kurtosis(abs(self))

    def mean(self):
        """
        Compute the spectral mean (first spectral moment)
        """
        return numpy.sum(abs(self)) / len(self)

    def mfcc(self, m, NumFilters = 48):
        """
        Compute the Mth Mel-Frequency Cepstral Coefficient
        """
        return MFCC.mfcc(self, m, NumFilters)

    def mfcc2(self, numFilters = 32):
        """
        Vectorized MFCC implementation
        """
        return MFCC.mfcc2(self, numFilters)

    def plot(self):
        """
        Plot the spectrum using matplotlib
        """
        plt.plot(abs(self))
        plt.xlim(0, len(self))
        plt.show()

    def rolloff(self):
        """
        Determine the spectral rolloff, i.e. the frequency below which 85% of the spectrum's energy
        is located
        """
        absSpectrum = abs(self)
        spectralSum = numpy.sum(absSpectrum)

        rolloffSum = 0
        rolloffIndex = 0
        for i in range(0, len(self)):
            rolloffSum = rolloffSum + absSpectrum[i]
            if rolloffSum > (0.85 * spectralSum):
                rolloffIndex = i
                break

        # Convert the index into a frequency
        frequency = rolloffIndex * (self.sampleRate / 2.0) / len(self)
        return frequency

    def skewness(self):
        """
        Compute the spectral skewness (third spectral moment)
        """
        return scipy.stats.skew(abs(self))

    def spread(self):
        """
        Compute the spectral spread (basically a variance of the spectrum around the spectral centroid)
        """
        centroid = self.centroid()

        binNumber = 0
        
        numerator = 0
        denominator = 0
        
        for bin in self:
            # Compute center frequency
            f = (self.sampleRate / 2.0) / len(self) 
            f = f * binNumber
            
            numerator = numerator + (((f - centroid) ** 2) * abs(bin))
            denominator = denominator + abs(bin)
            
            binNumber = binNumber + 1
            
        return math.sqrt((numerator * 1.0) / denominator)

    def variance(self):
        """
        Compute the spectral variance (second spectral moment)
        """
        return numpy.var(abs(self))