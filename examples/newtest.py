# newtest.py
# Test of new interfaces based on ndarray
import sys
sys.path.append('..')

from pymir import AudioFile

audiofile = AudioFile.open("../audio_files/test-stereo.mp3")
print "RMS: ", audiofile.rms()

spectrum = audiofile.spectrum()
print spectrum.sampleRate
