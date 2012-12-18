"""
MFCC methods
Compute Mel-Frequency Cepstral Coefficients
Last updated: 17 December 2012
"""

from __future__ import division

import math

import numpy
from numpy import *

import scipy
from scipy.fftpack import *

def mfcc2(spectrum, numFilters = 32):
	"""
	Alternative (and vectorized) MFCC computation from Steve Tjoa
	"""
	fb = filterbank(spectrum, spectrum.sampleRate, numFilters)
	coeff = scipy.fftpack.dct(scipy.log(fb), type = 2, norm = 'ortho')
	return coeff

def filterbank(x, fs, numFilters):
	n = len(x)
	m = 2 ** (1.0 / 6)
	f2 = 110.0
	f1 = f2 / m
	f3 = f2 * m
	fb = scipy.array(scipy.zeros(numFilters))
	for i in range(numFilters):
		fb[i] = numpy.absolute(fbwin(x, fs, f1, f2, f3))
		f1 = f2
		f2 = f3
 		f3 = f3 * m
	
	return fb

def fbwin(x, fs, f1, f2, f3):
	n = len(x)
	b1 = int(n * f1 / fs)
	b2 = int(n * f2 / fs)
	b3 = int(n * f3 / fs)
	y = x[b2]
	
	for b in range(b1, b2):
		y = y + x[b] * (b - b1) / (b2 - b1)
	
	for b in range(b2 + 1, b3):
		y = y + x[b] * (1 - (b - b2) / (b3 - b2))
	
	return y

def mfcc(spectrum, m, NumFilters = 48):
    """
    Compute the Mth Mel-Frequency Cepstral Coefficient
    """
    result = 0
    outerSum = 0
    innerSum = 0

    binSize = len(spectrum)

    if m >= NumFilters:
        return 0    # This represents an error condition - the specified coefficient is greater than or equal to the number of filters. The behavior in this case is undefined.

    result = normalizationFactor(NumFilters, m)

    for filterBand in range(1, NumFilters + 1):
        # Compute inner sum
        innerSum = 0
        for frequencyBand in range(0, binSize - 1):
            innerSum = innerSum + abs(spectrum[frequencyBand] * filterParameter(binSize, frequencyBand, filterBand, spectrum.sampleRate))

        if innerSum > 0:
            innerSum = log(innerSum) # The log of 0 is undefined, so don't use it

        innerSum = innerSum * math.cos(((m * math.pi) / NumFilters) * (filterBand - 0.5))
        
        outerSum = outerSum + innerSum

    result = result * outerSum

    return result

def normalizationFactor(NumFilters, m):
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

def filterParameter(binSize, frequencyBand, filterBand, samplingRate):
    """
    Intermediate computation used by the mfcc function. 
    Compute the filter parameter for the specified frequency and filter bands
    """
    filterParameter = 0
    boundary = (frequencyBand * samplingRate) / float(binSize) # k * Fs / N
    prevCenterFrequency = getCenterFrequency(filterBand - 1)   # fc(l - 1)
    thisCenterFrequency = getCenterFrequency(filterBand)       # fc(l)
    nextCenterFrequency = getCenterFrequency(filterBand + 1)   # fc(l + 1)

    if boundary >= 0 and boundary < prevCenterFrequency:
        filterParameter = 0
    
    elif boundary >= prevCenterFrequency and boundary < thisCenterFrequency:
        filterParameter = (boundary - prevCenterFrequency) / (thisCenterFrequency - prevCenterFrequency)
        filterParameter = filterParameter * getMagnitudeFactor(filterBand)
    
    elif boundary >= thisCenterFrequency and boundary < nextCenterFrequency:
            filterParameter = (boundary - nextCenterFrequency) / (thisCenterFrequency - nextCenterFrequency)
            filterParameter = filterParameter * getMagnitudeFactor(filterBand)
    
    elif boundary >= nextCenterFrequency and boundary < samplingRate:
            filterParameter = 0

    return filterParameter

def getMagnitudeFactor(filterBand):
    """
    Intermediate computation used by the mfcc function. 
    Compute the band-dependent magnitude factor for the given filter band
    """
    magnitudeFactor = 0
    
    if filterBand >= 1 and filterBand <= 14:
        magnitudeFactor = 0.015
    elif filterBand >= 15 and filterBand <= 48:
        magnitudeFactor = 2.0 / (getCenterFrequency(filterBand + 1) - getCenterFrequency(filterBand - 1))

    return magnitudeFactor

def getCenterFrequency(filterBand):
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