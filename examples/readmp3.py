# Example showing how to read mp3 files
import sys
sys.path.append('..')
from pymir.audio import mp3
from pymir.audio.transform import *
a = mp3.load("../audio_files/test-stereo.mp3")
t=stft(a)
a=istft(t)
mp3.play(a)
