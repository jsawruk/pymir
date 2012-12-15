"""
Transforms for converting between time and spectral domains
Includes: FFT/IFFT, DCT/IDCT, CQT
Last updated: 15 December 2012
"""

import numpy
import numpy.fft
from numpy import *

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

def idct(spectrum):
	"""
	Compute the Inverse Discrete Cosine Transform (IDCT)
 	"""
	N = len(spectrum)
	x = array(zeros(N))
	a = sqrt(2 / float(N))
	for n in range(N):
		for k in range(N):
			if k == 0:
				x[n] += sqrt(1 / float(N)) * spectrum[k] * cos(pi * (2 * n + 1) * k / float(2 * N))
			else:
				x[n] += a * spectrum[k] * cos(pi * (2 * n + 1) * k / float(2 * N))
	return x

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