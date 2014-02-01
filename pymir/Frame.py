"""
Frame class
ndarray subclass for time-series data
Last updated: 31 January 2014
"""
import math
from math import *

import numpy
import numpy.fft
from numpy import *
from numpy.lib import stride_tricks

import scipy

import matplotlib.pyplot as plt

import pymir
from pymir import Spectrum, Transforms
import pyaudio

class Frame(numpy.ndarray):
    
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
          strides=None, order=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = numpy.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)
        
        obj.sampleRate = 0
        obj.channels = 1
        obj.format = pyaudio.paFloat32
        
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        
        self.sampleRate = getattr(obj, 'sampleRate', None)
        self.channels = getattr(obj, 'channels', None)
        self.format = getattr(obj, 'format', None)
        
        # We do not need to return anything
        
    #####################
    # Frame methods
    #####################
    
    def cqt(self):
        """
        Compute the Constant Q Transform (CQT)
        """
        return Transforms.cqt(self)
    
    def dct(self):
        """
        Compute the Discrete Cosine Transform (DCT)
        """
        return Transforms.dct(self)
    
    def energy(self, windowSize = 256):
        """
        Compute the energy of this frame
        """
        N = len(self)

        window = numpy.hamming(windowSize)
        window.shape = (windowSize, 1)
        
        n = N - windowSize #number of windowed samples.
    
        # Create a view of signal who's shape is (n, windowSize). Use stride_tricks such that each stide jumps only one item.
        p = numpy.power(self,2)
        s = stride_tricks.as_strided(p,shape=(n, windowSize), strides=(self.itemsize, self.itemsize))
        e = numpy.dot(s, window) / windowSize
        e.shape = (e.shape[0], )
        return e
    
    def frames(self, frameSize, windowFunction = None):
        """
        Decompose this frame into smaller frames of size frameSize
        """
        frames = []
        start = 0
        end = frameSize
        while start < len(self):
            
            if windowFunction == None:
                frames.append(self[start:end])
            else:
                window = windowFunction(frameSize)
                window.shape = (frameSize, 1)
                window =  numpy.squeeze(window)
                frame = self[start:end]
                if len(frame) < len(window):
                    # Zero pad
                    frameType = frame.__class__.__name__

                    sampleRate = frame.sampleRate
                    channels = frame.channels
                    format = frame.format

                    diff = len(window) - len(frame)
                    frame = numpy.append(frame, [0] * diff)
                    
                    if frameType == "AudioFile":
                        frame = frame.view(pymir.AudioFile)
                    else:
                        frame = frame.view(Frame)

                    # Restore frame properties
                    frame.sampleRate = sampleRate
                    frame.channels = channels
                    frame.format = format
       
                windowedFrame = frame * window
                frames.append(windowedFrame)

            start = start + frameSize
            end = end + frameSize
            
        return frames
    
    def framesFromOnsets(self, onsets):
        """
        Decompose into frames based on onset start time-series
        """
        frames = []
        for i in range(0, len(onsets) - 1):
            frames.append(self[onsets[i] : onsets[i + 1]])

        return frames

    def play(self):
        """
        Play this frame through the default playback device using pyaudio (PortAudio)
        Note: This is a blocking operation.
        """
        # Create the stream
        p = pyaudio.PyAudio()
        stream = p.open(format = self.format, channels = self.channels, rate = self.sampleRate, output = True)

        # Write the audio data to the stream
        audioData = self.tostring()
        stream.write(audioData)

        # Close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    def plot(self):
        """
        Plot the frame using matplotlib
        """
        plt.plot(self)
        plt.xlim(0, len(self))
        plt.ylim(-1.5, 1.5)
        plt.show()

    def rms(self):
        """
        Compute the root-mean-squared amplitude
        """
        sum = 0
        for i in range(0, len(self)):
            sum = sum + self[i] ** 2
            
        sum = sum / (1.0 * len(self))
        
        return math.sqrt(sum)
    
    # Spectrum
    def spectrum(self):
        """
        Compute the spectrum using an FFT
        Returns an instance of Spectrum
        """
        return Transforms.fft(self)

    def zcr(self):
        """
        Compute the Zero-crossing rate (ZCR)
        """
        zcr = 0
        for i in range(1, len(self)):
            if (self[i - 1] * self[i]) < 0:
                zcr = zcr + 1
                
        return zcr / (1.0 * len(self))