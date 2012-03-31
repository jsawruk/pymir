# AudioFile.py
# Python class for reading in audio files

from pymir.audio import mp3
from pymir.audio import onset
from pymir.audio import Frame

import scipy.io.wavfile
import numpy

class AudioFile:
    
    data = []
    frames = []
    sampleRate = 44100
    
    # Read an MP3 file into memory
    def readMp3(self, filename, len=-1):
        self.data = mp3.load(filename)
        if len != -1:
            # Use only a portion of the audio
            self.data = self.data[:len]
    
    def readWav(self, filename, len=-1):
        sample_rate, samples = scipy.io.wavfile.read(filename)
        
        # Convert to mono
        samples = numpy.mean(samples, 1)
        
        self.data = samples
        
        if len != -1:
            # Use only a portion of the audio
            self.data = self.data[:len]
    
    def getOnsets(self):
        os = onset.onsets(self.data)
        
        start = 0
        for onsetTime in os:
            end = onsetTime
    
            startTime = start * (1.0 / self.sampleRate)
            endTime = end * (1.0 / self.sampleRate)
            
            frame = Frame.Frame(self.data[start:end], startTime, endTime, self.sampleRate)
    
            self.frames.append(frame)
    
            start = end
        
        # Append the last frame
        startTime = end * (1.0 / self.sampleRate)
        endTime = len(self.data) * (1.0 / self.sampleRate)
        frame = Frame.Frame(self.data[end:], startTime, endTime, self.sampleRate)
        self.frames.append(frame)