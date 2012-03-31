# Spectrum.py
# Store spectral data and compute spectral features

from pymir.audio import chroma

class Spectrum:
    data = []
    
    def __init__(self, data):
        self.data = data
        
    def getChroma(self):
        return chroma.chroma(self.data)