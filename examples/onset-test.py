# onset-test.py
# Testing of spectral flux and flux-based onset detection methods

import sys
sys.path.append('..')

from pymir import AudioFile
from pymir import SpectralFlux

import matplotlib.pyplot as plt

audiofile = AudioFile.open("../audio_files/test-stereo.mp3")

# Decompose the audio file into a set of frames of framesize
frameSize = 1024
frames = audiofile.frames(frameSize)

# Compute the spectra of each frame
spectra = [f.spectrum() for f in frames]

# Compute the spectral flux
flux = SpectralFlux.spectralFlux(spectra, rectify=True)
print flux
#plt.plot(flux)
#plt.show()