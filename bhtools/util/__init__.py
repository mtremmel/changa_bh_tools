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

def cutdict(target, goodinds):
    for key in list(target.keys()):
        target[key] = target[key][goodinds]
    return

def wrap(relpos,scale,boxsize=25e3):
    '''
    take an array of "raw" relative positions and update their values
    assuming a periodic box of size boxsize.
    :param relpos: the relative positions (dx,dy,dz) in physical units
    :param scale: the scale factor 1/(1+redshift)
    :param boxsize: the boxsize in co-moving kpc
    '''
    bphys = boxsize*scale
    bad = np.where(np.abs(relpos) > bphys/2.)
    if isinstance(bphys, np.ndarray):
        relpos[bad] = -1.0 * (relpos[bad] / np.abs(relpos[bad])) * np.abs(bphys[bad] - np.abs(relpos[bad]))
    else:
        relpos[bad] = -1.0 * (relpos[bad]/np.abs(relpos[bad])) * np.abs(bphys - np.abs(relpos[bad]))
    return

def write_dict_to_file(filename, my_dict, *keys_to_output_in_order, fmt=None):
    '''
    print out a dictionary of items to a file by providing the order of the keys
    :param filename: name of file to write to
    :param my_dict: dictionary object you want to write to a file
    :param keys_to_output_in_order: keys you want to output to file (in column order)
    :param fmt: format you want (e.g. fmt = %f, fmt = ['%d', '%f'])
    '''
    tofile = []
    for key in keys_to_output_in_order:
        tofile.append(my_dict[key])
    np.savetxt(filename, np.column_stack(tofile), fmt=fmt)