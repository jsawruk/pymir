# chords.py
# Chord estimator from MP3 file
import sys

import matplotlib.pyplot as plt

sys.path.append('..')
from pymir.audio import chordestimator

from pymir.audio import AudioFile

# Load the audio
audioFile = AudioFile.AudioFile()
audioFile.readMp3("../audio_files/test-stereo.mp3", 300000) # Only read in 100,000 samples

audioFile.getOnsets()

for frame in audioFile.frames:
    spectrum = frame.getSpectrum()
    chroma = spectrum.getChroma()
    chord = chordestimator.getChord(chroma)
    print frame.startTime, frame.endTime, chord
    