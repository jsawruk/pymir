# Compute chroma from a spectrum

import numpy
import math

def chroma(spectrum, fs = 44100):
    chroma = [0] * 12
    for index in range(0, len(spectrum)):
        
        # Assign a frequency value to each bin
        f = index * (fs / 2.0) / len(spectrum)
        
        # Convert frequency to pitch to pitch class
        if f != 0:
            pitch = int(round(69 + 12 * math.log(f / 440.0, 2)))
        else:
            pitch = 0
        pitchClass = pitch % 12
        
        chroma[pitchClass] = chroma[pitchClass] + abs(spectrum[index])
    
    # Normalize the chroma vector
    maxElement = max(chroma)
    chroma = [c / maxElement for c in chroma]
    
    return chroma