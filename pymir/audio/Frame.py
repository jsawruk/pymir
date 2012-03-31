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
    
    # Compute the zero-crossing rate
    def getZCR(self):
        zcr = 0
        for i in range(1, len(self.data)):
            if (self.data[i - 1] * self.data[i]) < 0:
                zcr = zcr + 1
                
        return zcr / (1.0 * len(self.data))