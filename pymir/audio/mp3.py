# MP3.py
# Functions for decoding MP3 into PCM using FFmpeg

import os
from subprocess import Popen, PIPE

import numpy
from numpy import *

def load(mp3File):
    sampleRate = 44100

    ffmpeg = Popen([
            "ffmpeg",
            "-i", mp3File,
            "-vn", "-acodec", "pcm_s16le", # Little Endian 16 bit PCM
            "-ac", "1", "-ar", str(sampleRate), # -ac = audio channels (1)
            "-f", "s16le", "-"], # -f wav for WAV file
            stdin = PIPE, stdout = PIPE, stderr = open(os.devnull, "w"))

    rawData =  ffmpeg.stdout

    mp3Array = numpy.fromstring(rawData.read(),numpy.int16)
    mp3Array = mp3Array.astype('float')/32767.0        
    #print mp3Array.size
    return mp3Array

def play(mp3Array):
	import scikits.audiolab
	scikits.audiolab.play(mp3Array)

