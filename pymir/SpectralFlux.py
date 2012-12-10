"""
SpectralFlux.py
Compute the spectral flux between consecutive spectra
This technique can be for onset detection

rectify - only return positive values
"""
def spectralFlux(spectra, rectify=False):
    """
    Compute the spectral flux between consecutive spectra
    """
    spectralFlux = []
    
    # Compute flux for zeroth spectrum
    flux = 0
    for bin in spectra[0]:
        flux = flux + abs(bin)
      
    spectralFlux.append(flux)
    
    # Compute flux for subsequent spectra
    for s in range(1, len(spectra)):
        prevSpectrum = spectra[s - 1]
        spectrum = spectra[s]
        
        flux = 0
        for bin in range(0, len(spectrum)):
            diff = abs(spectrum[bin]) - abs(prevSpectrum[bin])
            
            # If rectify is specified, only return positive values
            if rectify and diff < 0:
                diff = 0
            
            flux = flux + diff
            
        spectralFlux.append(flux)
        
    return spectralFlux