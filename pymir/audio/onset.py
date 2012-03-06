# Onset detector

import numpy
import scipy

from pymir.audio import energy

def onsets(signal):
    
    # Get energy
    e = energy.energy(signal)
    
    # Compute dEnergy
    windowSize = 256
    window = numpy.hamming(windowSize)
    
    #diffe = numpy.diff(e)
    
    #de = scipy.zeros(len(diffe))
    
    #for i in (0, len(diffe) - windowSize):
    #  de[i] = numpy.sum(window.conj().transpose() * diffe[i:i + windowSize]) / windowSize
    #e = [x + 0.00001 for x in e]
    e = e + 0.00001
    diffloge = numpy.diff(numpy.log(e)) 
          
    return diffloge