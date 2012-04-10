#!/usr/bin/env python
#-*- coding:utf-8 -*-
"""
mir.py: A simple, single-file package of music information retrieval
utilities for Python that emphasizes simplicity, modularity, and
visualizations.

Authors: Brennan Keegan, Steve Tjoa
Institution: Signals and Information Group, University of Maryland
Created: February 2, 2011
Last Modified: March 3, 2011

Basic functions:
wavread, micread, play
spectrogram, chromagram, instrogram
"""

import scipy
import pylab
import scipy.signal as ss
import scipy.linalg as sl
import scikits.audiolab as audiolab
import alsaaudio
import time
import os
import dct


### Audio I/O utilities ###

def wavread(filename):
    """
    wav, fs, nbits = wavread(filename)

    Read file FILENAME. WAV is a numpy array, FS is the sampling rate,
    and NBITS is the bit depth.
    """
    return audiolab.wavread(filename)

def micread(sec, fs=44100):
    """
    wav = micread(sec, fs=44100)

    Reads SEC seconds from the microphone at sampling rate FS.
    """
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK)
    inp.setchannels(1)
    inp.setrate(fs)
    inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    inp.setperiodsize(160)

    f = open('out.raw', 'wb')
    t0 = time.time()
    while time.time()-t0 < sec:
        l, data = inp.read()
        if l:
            f.write(data)

    x = scipy.fromfile('out.raw', dtype=scipy.int16)
    f.close()
    os.remove('out.raw')
    return x*2.0**-15

def play(wav, fs=44100):
    """
    play(wav, fs=44100)

    Play WAV at sampling rate FS.
    """
    audiolab.play(wav.T, fs)




### Gram class and subclasses ###

class Gram(object):

    def __init__(self, x, fs, framesz, hop, ch, func, tmin, tmax):
        self.fs = fs
        self.framesz = framesz
        self.hop = hop

        if x.ndim > 1:
            numch = x.shape[1]
            x = x[:,ch].squeeze()
        framesamp = int(fs*framesz)
        hopsamp = int(fs*hop)

        if tmin is None:
            self.tmin = 0
        else:
            self.tmin = tmin
        if tmax is None:
            self.tmax = float(x.shape[0])/fs
        else:
            self.tmax = tmax

        tminsamp = int(fs*self.tmin)
        tmaxsamp = int(fs*self.tmax)
        numsamp = tmaxsamp-tminsamp
        self.numframes = (numsamp-framesamp)/hopsamp + 1


        if func == scipy.fft:
            ymax = framesamp
        elif func == pc:
            ymax = int(hz2midi(fs/2))
        elif func == chroma:
            ymax = 12
        elif func == mfcc_fake:
            ymax = 32

        self.X = scipy.zeros([ymax, self.numframes])
        
        n = 0
        for i in range(tminsamp, tmaxsamp-framesamp, hopsamp):
            self.X[:,n] = func(x[i:i+framesamp])
            n = n + 1


    def plot(self, color=True, extent=None, ymax=None):
        #pylab.rcParams['text.usetex'] = True
        #pylab.rcParams['font.family'] = 'serif'

        if color:
            cm = None
        else:
            cm = pylab.cm.gray_r

        if ymax == None:
            img = pylab.imshow(self.X, aspect='auto', interpolation = 'nearest', origin='lower', extent=extent, cmap=cm)        
        else:
            x = self.X.shape
            smax = int(float(ymax*x[0])/self.fs)
            Xtrunc = self.X[0:smax,]
            img = pylab.imshow(Xtrunc, aspect='auto', interpolation = 'nearest', origin='lower', extent=extent, cmap=cm)

        pylab.show()
        pylab.xlabel('Time (seconds)')
        return img

def spectrogram(wav, fs, framesz, hop, ch=0, absolute=True, half=True, tmin=None, tmax=None):
    return Spectrogram(wav, fs, framesz, hop, ch, absolute, half, tmin, tmax)

class Spectrogram(Gram):

    def __init__ (self, x, fs, framesz, hop, ch, absolute, half, tmin, tmax):
        Gram.__init__(self, x, fs, framesz, hop, ch, scipy.fft, tmin, tmax)
        if absolute:
            self.X = scipy.absolute(self.X)
        if half:
            self.X = self.X[:self.X.shape[0]/2,:]

    def plot(self, title='Spectrogram', color=True, show=True, filename=None, fmax=None):
        if fmax == None:
            fmax = self.fs
        img = Gram.plot(self, color, extent=[self.tmin, self.tmax, 0, fmax], ymax = fmax)
        pylab.title(title)
        pylab.ylabel('Frequency (Hertz)')
        return img

def qspectrogram(wav, fs, framesz, hop, ch=0, absolute=True, half=False, tmin=None, tmax=None):
    return Qspectrogram(wav, fs, framesz, hop, ch, absolute, half, tmin, tmax)


class Qspectrogram(Gram):
    """

    """
    def __init__(self, x, fs, framesz, hop, ch, absolute, half, tmin, tmax):
        Gram.__init__(self, x, fs, framesz, hop, ch, pc, tmin, tmax)

        if absolute:
            self.X = scipy.absolute(self.X)
        if half:
            self.X = self.X[:self.X.shape[0]/2,]
    
    def plot(self, title='Constant Q Spectrogram', color=True, show=True, filename=None, fmax=None):
        img = Gram.plot(self, color, extent=[self.tmin, self.tmax, 0, int(hz2midi(self.fs/2))], ymax = fmax)
        pylab.title(title)
        pylab.ylabel('MIDI Note #')
        return img

def cqt(wav, fs=44100, lo=12, hi=None):
    """
    q, p = cqt(wav, fs, lo, hi)

    Returns the complex constant-Q transform vector of WAV with
    sampling rate FS. Like chroma, but unwrapped.
    Here, Q = 16.817.
    """    
    if wav.ndim==2:
        wav = wav[:,0].squeeze()
    wav *= ss.hamming(wav.size)
    X = scipy.fft(wav,n=2**15)
    return cqtfft(X, fs, lo, hi)
    

def cqtfft(X, fs=44100, lo=12, hi=None):
    """
    q, p = cqtfft(X, fs)

    Returns the complex constant-Q transform vector of fft vector X
    with sampling rate FS. Like chroma, but unwrapped.
    Here, Q = 16.817.
    """    
    if X.ndim==2:
        X = X[:,0].squeeze()
    if hi is None:
        hi = int(hz2midi(fs/2))
    p = scipy.arange(lo, hi)
    q = scipy.zeros(hi-lo)
    
    for i,midi in enumerate(p):
        center = X.size*midi2hz(midi)/fs
        width = int(center/16.817)
        center = int(center)
        w = ss.triang(2*width+1)
        q[i] = scipy.inner(w, X[center-width:center+width+1])

    return q, p

def cqtgram(x, fs, framesz, hop, lo=12, hi=None, absolute=True):
    hopsamp = int(hop*fs)
    framesamp = int(framesz*fs)
    Q = scipy.array([cqt(x[i:i+framesamp], fs, lo, hi)[0] for i in range(0, x.size-framesamp, hopsamp)]).T
    if absolute:
        Q = scipy.absolute(Q)
    return Q

def chromagram(wav, fs, framesz, hop):
    hopsamp = int(hop*fs)
    framesamp = int(framesz*fs)
    return scipy.array([chroma(wav[i:i+framesamp], fs) for i in range(0, wav.size-framesamp, hopsamp)]).T

class Chromagram(Gram):
    def __init__ (self,wav,fs,framesz,hop,ch,absolute,half,tmin,tmax):
        if wav.ndim==2:
            wav = wav[:,0].squeeze()
        wavlen = wav.size
        if wavlen > tmax*fs:
            wavlen = tmax*fs
        framesz_samp = int(framesz*fs)
        hop_samp = int(hop*fs)
        numframes = (wavlen-framesz_samp)/hop_samp + 1
        C = scipy.zeros([12, numframes])
        n = 0
        for k in range(numframes):
            C[:,k] = chroma(wav[n:n+framesz_samp]*ss.hamming(framesz_samp), fs)
            n += hop_samp
        self.C = C    

    def plot(self, title='Chromagram', color=True, show=True, filename=None):
        pylab.clf()
        pylab.imshow(self.C, origin='lower', interpolation='nearest', aspect='auto')
        pylab.yticks(range(12), labels())
        
        pylab.colorbar(format='%.2f')
        pylab.title(title)
        pylab.ylabel('Pitch')
        if show:
            pylab.show()
        if filename:
            pylab.savefig(filename)
        
def chroma(wav, fs):
    """
    Return chroma vector of a WAV vector with sampling rate FS.
    Assumes single-column vector input.
    """
    q, p = cqt(wav, fs, lo=12, hi=None)
    ch = scipy.zeros(12)
    for i in range(12):
        ch[i] = scipy.absolute(q[p%12==i]).sum()
    return ch/ch.sum()






def timbregram(x, fs, framesz,hop, ch=0, tmin=0,tmax=None):
    return Timbregram(x, fs, framesz, hop, ch, tmin, tmax)

class Timbregram(Gram):

    def __init__(self, x, fs, framesz, hop, ch, tmin, tmax):
        Gram.__init__(self, x, fs, framesz, hop, ch, mfcc_fake, tmin, tmax)
        self.Coeff_Num = 32

    def plot(self, title='Timbregram', color=True, show=True, filename=None, ymax=None):
        img = Gram.plot(self, color, extent=[self.tmin, self.tmax, 0, ymax], ymax = ymax)
        pylab.title(title)
        pylab.ylabel('MFCC Coefficients')
        return img
    
def mfcc_fake(x, fs=44100):
    c = numpy.arange(32)
    return c

class Scape(object):
 
    def __init__(self, x, fs, fsz_min, fsz_max, ch, func, tmin, tmax):

        self.fs = fs
        self.fsz_min = fsz_min
        self.fsz_max = fsz_max
        
        if fsz_max is None:
            self.fsz_max = float(len(x))/self.fs
        
        if x.ndim > 1:
            numch = x.shape[1]
            x = x[:,ch]
            
        fsamp_min = int(fs*self.fsz_min)
        fsamp_min = len(x)/(len(x)/fsamp_min)
        fsamp_max = len(x)
        
        self.fsamp_min = fsamp_min
        self.fsamp_max = fsamp_max
        

        if tmin is None:
            self.tmin = 0
        else:
            self.tmin = tmin
        
        if tmax is None:
            self.tmax = float(len(x))/self.fs
        else:
            self.tmax = tmax

        tminsamp = int(fs*self.tmin)
        tmaxsamp = int(fs*self.tmax)
        numsamp = tmaxsamp - tminsamp
        self.maxnumframes = numsamp/fsamp_min

        step =[]
        for i in range(1,len(x)):
            if (self.maxnumframes % i*fsamp_min) < 2*fsamp_min:
                step.append(i*fsamp_min)

        inc = len(step)
        self.inc = inc
        self.step = step
        self.steplabels = [self.fsz_min*float(i)/self.fsamp_min for i in step]
        self.X = numpy.zeros([inc,self.maxnumframes])
        
        n = 0        
        for fsamp in step:
            m = 0
            for i in range(tminsamp, tmaxsamp - fsamp, fsamp):
                self.X[n,m:(m+fsamp/fsamp_min)] = func(x[i:i+fsamp])
                m += fsamp/fsamp_min
            n += 1


    def plot(self, color=True, extent=None, ymax=None):

        if color:
            cm = None
        else:
            cm = pylab.cm.gray_r

        img = pylab.imshow(self.X, aspect='auto', interpolation = 'nearest', origin='upper', extent=extent, cmap=cm)        

        pylab.show()
        pylab.xlabel('Time (seconds)')
        return img

def keyscape(x, fs, fsz_min=None, fsz_max=None, ch=None, tmin=None, tmax=None):
    return Keyscape(x, fs, fsz_min, fsz_max, ch, tmin, tmax)


class Keyscape(Scape):

    def __init__ (self, x, fs, fsz_min, fsz_max, ch, tmin, tmax):
        Scape.__init__(self, x, fs, fsz_min, fsz_max, ch, Key, tmin, tmax)

    def plot(self, title='Keyscape', color=True, show=True, filename=None, ymax=None):
        if ymax == None:
            ymax = self.inc
        img = Scape.plot(self, color, extent=[self.tmin, self.tmax, 0, ymax], ymax = ymax)
        pylab.yticks(self.steplabels)
        pylab.title(title)
        pylab.ylabel('Resolution')
        return img

def Key(wav, fs = 44100):
    x = scipy.rand()

# Perform chroma analysis on signal, but afterwards sum all notes of a given type (store to different vector)

# Iterate 'key' template (W W H W W W H) over chroma output, multiplying notes in a given scale by 1,
# those not in the scale by 0, sum overall energy and use maximum outputs as possible keys

# Next, use original chroma matrix to look at first and last notes to check for major/minor,
# and to further decrease possibilities

# If multiple possibilities still remain, look for octave/P5 intervals in the interval coming after the first note,
# and in the first interval before the last note
    
    return x








### Feature extraction. ###



### Miscellaneous utilities. ###

def hz2midi(hz):
    """
    midi = hz2midi(hz)

    Converts frequency in Hertz to midi notation.
    """
    return 12*scipy.log2(hz/440.0) + 69

def midi2hz(midi):
    """
    hz = midi2hz(midi)

    Converts frequency in midi notation to Hertz.
    """
    return 440*2**((midi-69)/12.0)

def puretone(f0, T, fs=44100):
    """
    x, t = mir.puretone(f0, T, fs)

    Creates a pure tone at F0 Hertz, FS samples per second, for T seconds.
    Returns a numpy array X and its time index vector.
    >>>
    """
    t = scipy.arange(0,T,1.0/fs)
    return scipy.sin(2*scipy.pi*f0*t + scipy.rand()*scipy.pi*2), t

def labels(flat=False):
    if flat:
        return ['C',u'D\u266d','D',u'E\u266d','E','F',u'G\u266d','G',u'A\u266d','A',u'B\u266d','B']
    else:
        return ['C', 'C#','D','D#','E','F','F#','G','G#','A','A#','B']
        #return ['C',u'C\u266f','D',u'D\u266f','E','F',u'F\u266f','G',u'G\u266f','A',u'A\u266f','B']

def synpitch(midi, T, fs=44100):
    """
    Return a signal with pitch MIDI (in midi notation), duration T seconds,
    and sampling rate FS.
    """
    f0 = midi2hz(midi)
    f = f0
    P = 1
    x = puretone(f, fs, T)
    while f < fs/2:
        P *= 0.5
        f += f0
        x += P*puretone(f, fs, T)
    return x

def synpitchseq(notes, fs=44100):
    """
    NOTES is a list of tuples (MIDI, T). MIDI is the pitch (in midi notation)
    and T is the duration in seconds.
    """
    return scipy.concatenate([synpitch(midi, T, fs) for (midi, T) in notes])



def pc2key(c, pitches):
    """
    Given a pitch class vector, C, return the KEY associated.
    """
    key = 0
    for i, p in enumerate(pitches):
        if c[p-1] < c[p]:
            key += 1<<i
    return key

def pianoroll(pitches, phigh, show=True, figname=None):
    N = len(pitches)
    X = scipy.zeros([phigh, N])
    for i, frame in enumerate(pitches):
        for pitch in frame:
            X[pitch, i] = 1
    fig = pylab.figure(figsize=(16,4))
    ax = fig.add_axes([0.03,0.06,0.96,0.90])
    ax.imshow(X, origin='lower', aspect='auto', interpolation='nearest', cmap=pylab.cm.gray_r)
    ticks = range(12, phigh, 12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)
    ax.set_xticks([])
    ax.set_ylim([48, 84])
    if show:
        fig.show()
    if figname:
        fig.savefig(figname)



def mfcc(x, fs):
    """
    c = mfcc.mfcc(x, fs)
    Compute MFCCs.
    
    Inputs:
    x is a ndarray of a signal's FT magnitude (half: 0 to pi).
    fs is the sampling rate in Hertz.
        
    Author: Steve Tjoa
    Institution: University of Maryland
    Created: December 10, 2009
    Last modified: March 3, 2010
    """
    x = scipy.array(x).squeeze()
    fb = filterbank(x, fs)
    coeff = dct.dct(scipy.log(fb))
    return coeff
    
def filterbank(x, fs):
    n = len(x)
    m = 2**(1.0/6)
    f2 = 110.0
    f1 = f2/m
    f3 = f2*m
    fb = scipy.array(scipy.zeros(32))
    for i in range(32):
        fb[i] = fbwin(x, fs, f1, f2, f3)
        f1 = f2
        f2 = f3
        f3 = f3*m
    return fb
    
def fbwin(x, fs, f1, f2, f3):
    n = len(x)
    b1 = int(n*f1/fs)
    b2 = int(n*f2/fs)
    b3 = int(n*f3/fs)
    y = x[b2]
    for b in range(b1,b2):
        y = y + x[b]*(b - b1)/(b2 - b1)
    for b in range(b2+1, b3):
        y = y + x[b]*(1 - (b - b2)/(b3 - b2))
    return y

def pitch(x, fs, pitchrange=[12,120], mode='corr'):
    if mode=='corr':
        corr = scipy.correlate(x, x, mode='full')[len(x)-1:]
        corr[:int(fs/midi2hz(pitchrange[1]))] = 0
        corr[int(fs/midi2hz(pitchrange[0])):] = 0
        indmax = scipy.argmax(corr)
    elif mode=='ceps':
        y = rceps(x)
        y[:int(fs/midi2hz(pitchrange[1]))] = 0
        y[int(fs/midi2hz(pitchrange[0])):] = 0
        indmax = scipy.argmax(y)
    return hz2midi(fs/indmax)
        
    
def rceps(x):
    return scipy.real(scipy.ifft(scipy.log(scipy.absolute(scipy.fft(x)))))
    
def pitchceps(x, fs, pitchrange=[20,108]):
    y = rceps(x)
    return [y[int(fs/midi2hz(i))] for i in pitchrange]
