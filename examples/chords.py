"""
chords.py
Chord estimator from MP3 file
Last updated: 9 December 2012
"""
from __future__ import division 

import sys
sys.path.append('..')

from pymir import AudioFile
from pymir.audio import chordestimator
from pymir.audio import onsets

# Load the audio
print "Loading Audio"
audioFile = AudioFile.open("../audio_files/test-stereo.mp3")
audioFile = audioFile[:100000]

print "Extracting Frames"
frameSize = 16384
fixedFrames = audioFile.frames(frameSize)

print "Start | End  | Chord"
print "--------------------"

frameIndex = 0
for frame in fixedFrames:
    spectrum = frame.spectrum()
    chroma = spectrum.chroma()
    chord = chordestimator.getChord(chroma)
    startTime = (frameIndex * frameSize) / audioFile.sampleRate
    endTime = ((frameIndex + 1) * frameSize) / audioFile.sampleRate

    print "%.2f  | %.2f | %s" % (startTime, endTime, chord)
    frameIndex = frameIndex + 1