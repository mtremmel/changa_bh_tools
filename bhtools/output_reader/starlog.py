from ..util import *

def read_starlog(path_to_simulation='.', simname=None):
    starlogfile = find_file_by_extension('.starlog',path_to_simulation, simname)
    print(starlogfile)
    import pynbody
    print("reading starlog file...")
    sl = pynbody.snapshot.tipsy.StarLog(starlogfile)
    return sl