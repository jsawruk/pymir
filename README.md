# PyMIR
PyMIR is a Python library for common tasks in Music Information Retrieval (MIR)

## Prerequisites
* [Numpy](http://www.scipy.org/)
* [Scipy](http://www.scipy.org/)
* [FFmpeg executable](http://ffmpeg.org/)
* [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/) (for playback)
* [Matplotlib](http://matplotlib.org/) (for graphing)

## Features
* Read WAV files (using scipy) and MP3 files (using FFmpeg)
* Temporal feature extraction (Frame class)
    * Constant-Q Transform
    * Discrete Cosine Transform
    * Energy
    * Frame segmentation from onsets
    * RMS
    * Spectrum (FFT)
    * Zero-crossing rate
* Spectral feature extraction (Spectrum class)
    * Spectral Centroid
    * Spectral Flatness
    * Spectral Moments (mean, variance, skewness, kurtosis)
    * Spectral Spread
    * Spectral Rolloff
    * Spectral Crest Factor
    * Chroma
    * Inverse Discrete Cosine Transform
    * Inverse FFT
* Other features
    * Audio playback via PyAudio
    * Naive chord estimation
    * Naive pitch estimation
    * Onset detectors (energy, flux)
    * Spectral Flux

## Examples

The standard workflow for working with PyMIR is:
* Open an audio file (wav or mp3)
* Decompose into frames
    * Decomposed into fixed-size frames or
    * Use an onset detector
* Extract temporal features
* Extract spectral features

### Opening an audio file (AudioFile class)

    from pymir import AudioFile
    wavData = AudioFile.open("audio.wav")
    mp3Data = AudioFile.open("audio.mp3")

### Decomposing into frames

#### Fixed frame size
    fixedFrames = wavData.frames(1024)

    windowFunction = numpy.hamming
    fixedFrames = audiofile.frames(1024, windowFunction)

#### Using an onset detector
	from pymir.audio import onsets
	energyOnsets = onsets.onsetsByEnergy(wavData)
    framesFromOnsets = wavData.framesFromOnsets(energyOnsets)

### Extracting temporal features (Frame class)
    fixedFrames[0].cqt() 						# Constant Q Transform
    fixedFrames[0].dct() 						# Discrete Cosine Transform
    fixedFrames[0].energy(windowSize = 256) 	# Energy
    fixedFrames[0].play()                       # Playback using pyAudio
    fixedFrames[0].plot()                       # Plot using matplotlib
    fixedFrames[0].rms() 						# Root-mean-squared amplitude
    fixedFrames[0].zcr() 						# Zero-crossing raate

### Extracting spectral features
    # Compute the spectra of each frame
	spectra = [f.spectrum() for f in fixedFrames]
    spectra[0].centroid() 						# Spectral Centroid
    spectra[0].chroma()							# Chroma vector
    spectra[0].crest()                          # Spectral Crest Factor
    spectra[0].flatness()                       # Spectral Flatness
    spectra[0].idct()							# Inverse DCT
    spectra[0].ifft()							# Inverse FFT
    spectra[0].kurtosis()                       # Spectral Kurtosis
    spectra[0].mean()                           # Spectral Mean
    spectra[0].mfcc2()                          # MFCC (vectorized implementation)
    spectra[0].plot()                           # Plot using matplotlib
    spectra[0].rolloff()                        # Spectral Rolloff
    spectra[0].skewness()                       # Spectral Skewness
    spectra[0].spread()                         # Spectral Spread
    spectra[0].variance()                       # Spectral Variance

    from pymir import SpectralFlux

	# Compute the spectral flux
	flux = SpectralFlux.spectralFlux(spectra, rectify = True)

### Audio playback

Playback is provided on all AudioFile and Frame objects. Internal representation is 32-bit floating point.

    wavData.play()
    fixedFrames[0].play()

### Naive chord estimation

Naive chord estimation using a dictionary of the 24 major and minor triads only, represented as
normalized chroma vectors. Similarity is measured using the cosine similarity function. The closest
match is returned (as a string). 

This is called a naive approach because it does not consider preceding chords, which could improve
chord estimation accuracy.

To use, compute the chroma vector from a spectrum, and then use the getChord method

    spectrum = frame.spectrum()
    chroma = spectrum.chroma()
    chord = chordestimator.getChord(chroma)
