# Frame.py
# Represents a variable length time-domain frame

import numpy.fft
from pymir.audio import Spectrum

class Frame:
    data = []
    startTime = 0
    endTime = 0
    sampleRate = 44100
    
    def __init__(self, data, startTime, endTime, sampleRate):
        self.data = data
        self.startTime = startTime
        self.endTime = endTime
        self.sampleRate = sampleRate
        
    def getSpectrum(self):
        spectrum = numpy.fft.rfft(self.data)
        return Spectrum.Spectrum(spectrum)
        