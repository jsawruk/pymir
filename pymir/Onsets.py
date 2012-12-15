"""
Compute onset times from time-domain audio data
Spectra are computed as necessary
Supported methods:
- Time-domain: energy
- Spectral: flux

Last updated: 15 December 2012
"""
from pymir import Energy
from pymir import SpectralFlux

import numpy
from numpy import NaN, Inf, arange, isscalar, array, asarray

import matplotlib.pyplot as plt

def onsets(audioData, method='energy'):
	onsets = []
	if method == 'energy':
		onsets =  onsetsByEnergy(audioData)
	elif method == 'flux':
		onsets = onsetsByFlux(audioData)

	return onsets


def onsetsByEnergy(audioData, frameSize = 512, threshold = 1):
	"""
	Compute onsets by using dEnergy (time-domain)
	"""
	e = Energy.energy(audioData, frameSize)
	dE = Energy.dEnergy(audioData, frameSize)
	peaks = peakPicking(dE, 2048, threshold)

	return peaks

def onsetsByFlux(audioData, frameSize = 1024):
	"""
	Compute onsets by using spectral flux
	"""
	frames = audioData.frames(frameSize)

	# Compute the spectra of each frame
	spectra = [f.spectrum() for f in frames]

	# Compute the spectral flux
	flux = SpectralFlux.spectralFlux(spectra, rectify=True)

	peaks = peakPicking(flux, windowSize = 10, threshold = 1e6)
	peaks = [frameSize * p for p in peaks]

	return peaks

def peakPicking(onsets, windowSize = 1024, threshold = 1):

	peaks = []
	
	peaks = peaksAboveAverage(onsets, windowSize)

	# Compute a windowed (moving) average
	#movingAverage = windowedAverage(onsets, windowSize)

	#peaks = peakdet(movingAverage, 1, threshold = threshold)

	#for i in range(0, len(movingAverage) - 1):
	#	if movingAverage[i] > movingAverage[i + 1]:
	#		peaks.append(movingAverage[i])
	#	else:
	#		peaks.append(0)
	return peaks

def peaksAboveAverage(data, windowSize):
	"""
	Find peaks by the following method:
	- Compute the average of all the data
	- Using a non-sliding window, find the max within each window
	- If the windowed max is above the average, add it to peaks
	"""

	data = numpy.array(data)

	peaks = []

	dataAverage = numpy.average(data)
	dataAverage = dataAverage * 1

	slideAmount = windowSize / 2

	start = 0
	end = windowSize
	while start < len(data): 
		#print "Start: " + str(start)
		#print "End:   " + str(end)
		windowMax = data[start:end].max()  
		windowMaxPos = data[start:end].argmax()

		if windowMax > dataAverage:
			if (start + windowMaxPos) not in peaks:
				peaks.append(start + windowMaxPos)

		start = start + slideAmount
		end = end + slideAmount
	
	return peaks


def windowedAverage(data, windowSize):
	window = numpy.repeat(1.0, windowSize) / windowSize
	return numpy.convolve(data, window)[windowSize - 1 : -(windowSize - 1)]

def peakdet(v, delta, x = None, threshold = 1):
	"""
	Adapted from code at: https://gist.github.com/250860
	Converted from MATLAB script at http://billauer.co.il/peakdet.html

	Returns two arrays

	function [maxtab, mintab]=peakdet(v, delta, x)
	%PEAKDET Detect peaks in a vector
	%        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
	%        maxima and minima ("peaks") in the vector V.
	%        MAXTAB and MINTAB consists of two columns. Column 1
	%        contains indices in V, and column 2 the found values.
	%      
	%        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
	%        in MAXTAB and MINTAB are replaced with the corresponding
	%        X-values.
	%
	%        A point is considered a maximum peak if it has the maximal
	%        value, and was preceded (to the left) by a value lower by
	%        DELTA.

	% Eli Billauer, 3.4.05 (Explicitly not copyrighted).
	% This function is released to the public domain; Any use is allowed.

	"""
	maxtab = []
	mintab = []
	   
	if x is None:
	    x = arange(len(v))

	v = asarray(v)

	if len(v) != len(x):
	    sys.exit('Input vectors v and x must have same length')

	if not isscalar(delta):
	    sys.exit('Input argument delta must be a scalar')

	if delta <= 0:
	    sys.exit('Input argument delta must be positive')

	mn, mx = Inf, -Inf
	mnpos, mxpos = NaN, NaN

	lookformax = True

	for i in arange(len(v)):
	    this = v[i]
	    if this > mx:
	        mx = this
	        mxpos = x[i]
	    if this < mn:
	        mn = this
	        mnpos = x[i]
	    
	    if lookformax:
	        if this < mx - delta and this > threshold:
	            #maxtab.append((mxpos, mx))
	            maxtab.append(mxpos)
	            mn = this
	            mnpos = x[i]
	            lookformax = False
	    else:
	        if this > mn + delta:
	            #mintab.append((mnpos, mn))
	            mx = this
	            mxpos = x[i]
	            lookformax = True

	#return array(maxtab), array(mintab)
	return maxtab