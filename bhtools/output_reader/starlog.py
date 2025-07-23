def read_starlog(simname):
    import pynbody
    print("reading starlog file...")
    sl = pynbody.snapshot.tipsy.StarLog(simname+'.starlog')
    return sl