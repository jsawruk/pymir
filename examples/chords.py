"""
chords.py
Chord estimator from MP3 file
Last updated: 9 December 2012
"""
from __future__ import division 

import sys
sys.path.append('..')

from pymir import AudioFile
from pymir import Pitch
from pymir import Onsets

import matplotlib.pyplot as plt

# Load the audio
print "Loading Audio"
audiofile = AudioFile.open("../audio_files/test-stereo.mp3")

plt.plot(audiofile)
plt.show()

print "Finding onsets using Spectral Flux (spectral domain)"
o = Onsets.onsetsByFlux(audiofile)
print o

print "Extracting Frames"
frames = audiofile.framesFromOnsets(o)
#for i in range(0, len(frames)):
#	print "Frame " + str(i)
#	plt.plot(frames[i])
#	plt.show()

#frameSize = 16384
#frames = audioFile.frames(frameSize)

print "Start | End  | Chord | (% match)"
print "-------------------------------"

frameIndex = 0
startIndex = 0
for frame in frames:
	spectrum = frame.spectrum()
	chroma = spectrum.chroma()
	print chroma
	
	chord, score = Pitch.getChord(chroma)

	endIndex = startIndex + len(frame)

	startTime = startIndex / frame.sampleRate
	endTime = endIndex / frame.sampleRate

	print "%.2f  | %.2f | %-4s | (%.2f)" % (startTime, endTime, chord, score)
    
	frameIndex = frameIndex + 1
	startIndex = startIndex + len(frame)