"""
Transforms for converting between time and spectral domains
Includes: FFT/IFFT, DCT/IDCT, CQT
Last updated: 15 December 2012
"""

import numpy
import numpy.fft
from numpy import *

import scipy
from scipy.fftpack import *

import pymir

# Fourier Transforms
def fft(frame):
	"""
	Compute the spectrum using an FFT
	Returns an instance of Spectrum
	"""
	fftdata = numpy.fft.rfft(frame) # rfft only returns the real half of the FFT values, which is all we need
	spectrum = fftdata.view(pymir.Spectrum)
	spectrum.sampleRate = frame.sampleRate
	return spectrum

def ifft(spectrum):
	"""
	Compute the Inverse FFT
	"""
	fftdata = numpy.fft.irfft(spectrum)
	frame = fftdata.view(pymir.Frame)
	frame.sampleRate = spectrum.sampleRate
	return frame

# Discrete Cosine Transforms (DCT)
def dct(frame):
	"""
	Compute the Discrete Cosine Transform (DCT)
	"""
	dctResult = scipy.fftpack.dct(frame, type = 2, norm = 'ortho')
	dctSpectrum = dctResult.view(pymir.Spectrum)
	dctSpectrum.sampleRate = frame.sampleRate
	return dctSpectrum

def idct(spectrum):
	"""
	Compute the Inverse Discrete Cosine Transform (IDCT)
 	"""
	idctResult = scipy.fftpack.idct(spectrum, type = 2, norm = 'ortho')
	idctFrame = idctResult.view(pymir.Frame)
	idctFrame.sampleRate = spectrum.sampleRate
	return idctFrame

# Constant Q Transform
def cqt(frame):
	"""
	Compute the Constant Q Transform (CQT)
	"""
	N = len(frame)
	y = array(zeros(N))
	a = sqrt(2 / float(N))
	for k in range(N):
 		for n in range(N):
			y[k] += frame[n] * cos(pi * (2 * n + 1) * k / float(2 * N))
			
			if k == 0:
				y[k] = y[k] * sqrt(1 / float(N))
			else:
				y[k] = y[k] * a
	
	return y