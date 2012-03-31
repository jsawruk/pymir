# Read a wav file

import sys
sys.path.append('..')

from pymir.audio import AudioFile

# Load the audio
audioFile = AudioFile.AudioFile()
audioFile.readWav("../audio_files/test-stereo.wav", 100000) # Only read in 100,000 samples