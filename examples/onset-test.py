"""
Tests of different onset detection methods
Currently under development
Last updated: 9 December 2012
"""
import sys
sys.path.append('..')

from pymir import AudioFile
from pymir.audio import energy
from pymir.audio import onsets

import matplotlib.pyplot as plt

filename = "../audio_files/drum_loop_01.wav"

print "Opening File: " + filename
audiofile = AudioFile.open(filename)

# Time-based methods
print "Finding onsets using Energy function (temporal domain)"
o = onsets.onsetsByEnergy(audiofile)
print o
frames = audiofile.framesFromOnsets(o)

for i in range(0, len(frames)):
	print "Frame " + str(i)
	plt.plot(frames[i])
	plt.show()

# Spectral-based methods
print "Finding onsets using Spectral Flux (spectral domain)"
o = onsets.onsetsByFlux(audiofile)
print o
frames = audiofile.framesFromOnsets(o)
for i in range(0, len(frames)):
	print "Frame " + str(i)
	plt.plot(frames[i])
	plt.show()