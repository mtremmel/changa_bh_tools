from . import param_reader, readcol, cosmology
import numpy as np
import pynbody

#useful constants
phys_const = {
'c' : pynbody.array.SimArray(2.99792458e10, 'cm s**-1'),
'lbol_sun' : pynbody.array.SimArray(3.9e33, 'erg s**-1'),
'G' : pynbody.array.SimArray(6.67259e-8, 'cm**3 s**-2 g**-1'),
'M_sun_g' : pynbody.array.SimArray(1.988547e33,'g'),
'mh' : pynbody.array.SimArray(1.6726219e-24, 'g'),
'kb' : pynbody.array.SimArray(1.380658e-16, 'erg K**-1')
}

def ledd(mass):
    return mass * phys_const['lbol_sun'] * 3.2e4

def smoothdata(rawdat,nsteps=20,ret_std=False, dosum=False):
    nind = len(rawdat) - len(rawdat)%nsteps
    use = np.arange(nind)
    newdat = rawdat[use].reshape((int(nind/nsteps), nsteps))
    if dosum:
        meandat = np.nansum(newdat, axis=1)
    else:
        meandat = np.nanmean(newdat, axis=1)
    if ret_std is False:
        return meandat
    else:
        std = rawdat.std(axis=1)
        return meandat, std