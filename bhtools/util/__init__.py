from . import param_reader, readcol, cosmology
import numpy as np

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