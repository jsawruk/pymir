"""
Functions for tranforming audio signals (I.E. stft, mfcc, etc.)
stft and istft are written by Steve Tjoa and taken from http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
"""

import scipy, pylab

def stft(x, fs=44100, framesz=0.05, hop=0.020):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs=44100, T=5, hop=0.020):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

if __name__ == "__main__":
	import doctest
	doctest.testmod()
