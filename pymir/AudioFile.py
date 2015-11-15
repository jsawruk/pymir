"""
AudioFile class
Load audio files (wav or mp3) into ndarray subclass
Last updated: 15 December 2012
"""
import os
from subprocess import Popen, PIPE

import numpy
from numpy import *

import scipy.io.wavfile

from pymir import Frame
import pyaudio


class AudioFile(Frame.Frame):

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
        if obj is None:
            return
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

    @staticmethod
    def open(filename, sampleRate=44100):
        """
        Open a file (WAV or MP3), return instance of this class with data loaded in
        Note that this is a static method. This is the preferred method of constructing this object
        """
        _, ext = os.path.splitext(filename)


        if ext.endswith('mp3') or ext.endswith('m4a'):

            ffmpeg = Popen([
                "ffmpeg",
                "-i", filename,
                "-vn", "-acodec", "pcm_s16le",  # Little Endian 16 bit PCM
                "-ac", "1", "-ar", str(sampleRate),  # -ac = audio channels (1)
                "-f", "s16le", "-"],  # -f wav for WAV file
                stdin=PIPE, stdout=PIPE, stderr=open(os.devnull, "w"))

            rawData = ffmpeg.stdout

            mp3Array = numpy.fromstring(rawData.read(), numpy.int16)
            mp3Array = mp3Array.astype('float32') / 32767.0
            audioFile = mp3Array.view(AudioFile)

            audioFile.sampleRate = sampleRate
            audioFile.channels = 1
            audioFile.format = pyaudio.paFloat32

            return audioFile

        elif ext.endswith('wav'):
            sampleRate, samples = scipy.io.wavfile.read(filename)

            # Convert to float
            samples = samples.astype('float32') / 32767.0

            # Get left channel
            if len(samples.shape) > 1:
                samples = samples[:, 0]

            audioFile = samples.view(AudioFile)
            audioFile.sampleRate = sampleRate
            audioFile.channels = 1
            audioFile.format = pyaudio.paFloat32

            return audioFile
