"""
Onset detector

This will detect the beginning of musically interesting things, like notes and such.
"""
import numpy
import scipy

#from pymir.audio import energy

def onsets(signal,ws=2000,threshold=1.1,avg_coef=.999,min_dist=400):
	""" Calculates where the onsets are, and returns an array of sample numbers of where the onsets are """	
	os = list()
	af = 0.0
	sc = 0
	lat = -2*min_dist
	th = 0.005
	m = 0+abs(signal[1]-signal[0])
	for f in range(1,len(signal)):
		ds=abs(signal[f]-signal[f-1])
		m *= avg_coef
		 
		m += (1.0-avg_coef)*ds
		af = af + ds
		sc = sc + 1
		if sc > 2000:
			dso = abs(signal[f-ws]-signal[f-ws-1])
			af = af - dso
			sc = sc - 1
			if (af > threshold*m*ws):
				if f - lat > min_dist:
					for f2 in range(0,ws):
						if abs(signal[f-ws+f2]-signal[f-ws+f2-1])>threshold*m:
							os.append(f-ws+f2)
							break
				lat = f
	return(os)	
if __name__ == "__main__":
	from scikits.audiolab import Sndfile
	f = Sndfile("../../audio_files/beatbox1/callout_adiao.wav",'r')
	buf = f.read_frames(f.nframes, dtype=numpy.float64)
	os = onsets(buf)	
	for o in os:
		print str(float(o)/f.samplerate) 
