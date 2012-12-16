"""
Pitch functions
- Chroma
Last updated: 15 December 2012
"""
from __future__ import division

import math

import numpy

# Dictionary of major and minor chords
chords = [ {'name': "C", 'vector' :[1,0,0,0,1,0,0,1,0,0,0,0], 'key': 0, 'mode': 1 },
			{'name': "Cm", 'vector':[1,0,0,1,0,0,0,1,0,0,0,0], 'key': 0, 'mode': 0 },
			{'name': "C#", 'vector' :[0,1,0,0,0,1,0,0,1,0,0,0], 'key': 1, 'mode': 1 },
			{'name': "C#m", 'vector':[0,1,0,0,1,0,0,0,1,0,0,0], 'key': 1, 'mode': 0 },
			{'name': "D", 'vector' :[0,0,1,0,0,0,1,0,0,1,0,0],  'key': 2, 'mode': 1 },
			{'name': "Dm", 'vector':[0,0,1,0,0,1,0,0,0,1,0,0],  'key': 2, 'mode': 0 },
			{'name': "Eb", 'vector' :[0,0,0,1,0,0,0,1,0,0,1,0],  'key': 3, 'mode': 1 },
			{'name': "Ebm", 'vector':[0,0,0,1,0,0,1,0,0,0,1,0],  'key': 3, 'mode': 0 },
			{'name': "E", 'vector' :[0,0,0,0,1,0,0,0,1,0,0,1],  'key': 4, 'mode': 1 },
			{'name': "Em", 'vector':[0,0,0,0,1,0,0,1,0,0,0,1],  'key': 4, 'mode': 0 },
			{'name': "F", 'vector' :[1,0,0,0,0,1,0,0,0,1,0,0],  'key': 5, 'mode': 1 },
			{'name': "Fm", 'vector':[1,0,0,0,0,1,0,0,1,0,0,0],  'key': 5, 'mode': 0 },
			{'name': "F#", 'vector' :[0,1,0,0,0,0,1,0,0,0,1,0],  'key': 6, 'mode': 1 },
			{'name': "F#m", 'vector':[0,1,0,0,0,0,1,0,0,1,0,0],  'key': 6, 'mode': 0 },
			{'name': "G", 'vector' :[0,0,1,0,0,0,0,1,0,0,0,1],  'key': 7, 'mode': 1 },
			{'name': "Gm", 'vector':[0,0,1,0,0,0,0,1,0,0,1,0],  'key': 7, 'mode': 0 },
			{'name': "Ab", 'vector' :[1,0,0,1,0,0,0,0,1,0,0,0],  'key': 8, 'mode': 1 },
			{'name': "Abm", 'vector':[0,0,0,1,0,0,0,0,1,0,0,1],  'key': 8, 'mode': 0 },
			{'name': "A", 'vector' :[0,1,0,0,1,0,0,0,0,1,0,0],  'key': 9, 'mode': 1 },
			{'name': "Am", 'vector':[1,0,0,0,1,0,0,0,0,1,0,0],  'key': 9, 'mode': 0 },
			{'name': "Bb", 'vector' :[0,0,1,0,0,1,0,0,0,0,1,0],  'key': 10, 'mode': 1 },
			{'name': "Bbm", 'vector':[0,1,0,0,0,1,0,0,0,0,1,0],  'key': 10, 'mode': 0 },
			{'name': "B", 'vector' :[0,0,0,1,0,0,1,0,0,0,0,1],  'key': 11, 'mode': 1 },
			{'name': "Bm", 'vector':[0,0,1,0,0,0,1,0,0,0,0,1],  'key': 11, 'mode': 0 }]

def chroma(spectrum):
	"""
	Compute the 12-ET chroma vector from this spectrum
	"""
	chroma = [0] * 12
	for index in range(0, len(spectrum)):

		# Assign a frequency value to each bin
		f = index * (spectrum.sampleRate / 2.0) / len(spectrum)

		# Convert frequency to pitch to pitch class
		if f != 0:
			pitch = frequencyToMidi(f)
		else:
			pitch = 0
		pitchClass = pitch % 12

		chroma[pitchClass] = chroma[pitchClass] + abs(spectrum[index])

	# Normalize the chroma vector
	maxElement = max(chroma)
	chroma = [c / maxElement for c in chroma]
	
	return chroma

def cosineSimilarity(a, b):
	"""
	Compute the similarity between two vectors using the cosine similarity metric
	"""
	dotProduct = 0
	aMagnitude = 0
	bMagnitude = 0
	for i in range(len(a)):
		dotProduct += (a[i] * b[i])
		aMagnitude += math.pow(a[i], 2)
		bMagnitude += math.pow(b[i], 2)

	aMagnitude = math.sqrt(aMagnitude)
	bMagnitude = math.sqrt(bMagnitude)

	return dotProduct / (aMagnitude * bMagnitude)

def frequencyToMidi(frequency):
	"""
	Convert a given frequency in Hertz to its corresponding MIDI pitch number (60 = Middle C)
	"""
	return int(round(69 + 12 * math.log(frequency / 440.0, 2)))

def getChord(chroma):
	"""
	Given a chroma vector, return the best chord match using naive dictionary-based method
	"""
	maxScore = 0
	chordName = ""
	for chord in chords:
		score = cosineSimilarity(chroma, chord['vector'])
		if score > maxScore:
			maxScore = score
			chordName = chord['name']

	return chordName, maxScore

def naivePitch(spectrum):
	"""
	Compute the pitch by using the naive pitch estimation method, i.e. get the pitch name for the most
	prominent frequency.
	Only returns MIDI pitch number
	"""
	maxFrequencyIndex = numpy.argmax(spectrum)
	maxFrequency = maxFrequencyIndex * (spectrum.sampleRate / 2.0) / len(spectrum)
	return frequencyToMidi(maxFrequency)