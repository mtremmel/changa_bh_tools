from .output_reader import *
from .util import *
from .tangos_interface import *

class BHCatalog(object):
    def __init__(self, simname):
        self.simname=simname
        print("loading in orbit file...")
        self.orbitdata = orbit.BlackHoles(simname)
        print("loading in mergers file...")
        self.mergers = mergers.BHMergers(simname)
    
    def parameters(self):
        #get the loaded in parameters which shouls already have been loaded in with both orbit and mergers
        return self.orbitdata.parameters
	
    def __getitem__(self, item):
        if type(item)==str or (type(item)==tuple and len(item)==2):
            return self.orbitdata[item]
        
        if type(item)==tuple and len(item)==3:
            if item[2] not in ['raw','major']:
                raise ValueError("invalid arguments to __getitem__, use bhcatalog[(iord,'key','major'/'raw')]")
            if item[2] =='raw':
                return self.orbitdata[item[0],item[1]]
            if item[2] == 'major':
                select_id = item[0]
                select_data = item[1]
                if (type(select_id)!=np.int64 and type(select_id)!=int) or type(select_data)!=str:
                    raise ValueError("invalid arguments to orbit __getitem__, use bhcatalog[(iord,'key','major')]")
                t,iord = self._get_major_prog_nodes(select_id)
                slice = self.orbitdata._concatenate_iord_slices(select_id, iord, t)
                return self.orbitdata._data[select_data][slice]
            
    def _get_major_prog_nodes(self,bhid, dtmin=None,dmin=None, tstart=None):
        from .output_reader import mergers
        if tstart is None:
            tmax = self.orbitdata._data['time'][self.orbitdata._bhind[bhid]].max()
        else:
            tmax=tstart
        merge_id, tmerge, mass1, mass2, merge_id_all = \
			mergers.get_mergers_by_id(bhid, self.mergers, tmax, dmin=dmin, dtmin=dtmin)
        tord = np.argsort(tmerge)[::-1]
        tmerge = tmerge[tord]
        mass1 = mass1[tord]
        mass2 = mass2[tord]
        merge_id = merge_id[tord]
        nmerge = len(merge_id)
        i = 0
        cur_prog_id = bhid
        tbranch_change = []
        id_branch_change = []
        while i<nmerge:
            cur_merge_id = merge_id[i]
            if mass1[i]<mass2[i]:
                tbranch_change.append(tmerge[i])
                id_branch_change.append(cur_merge_id)
                cur_prog_id = cur_merge_id
                merge_id, tmerge, mass1, mass2, merge_id_all = \
					mergers.get_mergers_by_id(cur_prog_id, self.mergers, tmerge[i], dmin=dmin, dtmin=dtmin)
                nmerge = len(merge_id)
                if nmerge==0:
                    break
                tord = np.argsort(tmerge)[::-1]
                tmerge = tmerge[tord]
                mass1 = mass1[tord]
                mass2 = mass2[tord]
                merge_id = merge_id[tord]
                nmerge = len(merge_id)
            else:
                i+=1
        return tbranch_change, id_branch_change
    
    def time_smooth(self, iord, prop, track='major',nsmooth=10,dosum=False):
        data = smoothdata(self[iord,prop,track], nsteps=nsmooth, dosum=dosum)
        time = self[iord,'time',track][int(nsmooth/2)::nsmooth]
        while len(time)>len(data):
            print("timeslice too big, removing last entry")
            time = time[:-1]
        return data, time
    

    def smoothed_accretion_history(self, iord, dt='10 Myr', track='major'):
        if type(track)!=str or track not in ['major','raw']:
            raise ValueError("Keyword track must be either 'major' or 'raw'")
        if type(dt)==int or type(dt)==float:
            print("Received a number for keyword dt, assuming this corresonds to Myr")
            dt = str(dt)+' Myr'
        tsmooth = pynbody.units.Unit(dt)
        nsmooth = int(tsmooth.ratio(self.orbitdata.dTout))
        if track=='raw':
            dMacc, time = self.orbitdata.time_smooth(iord, 'dMtot', nsmooth=nsmooth, dosum=True)
            mdot_smooth = pynbody.array.SimArray(dMacc,'Msol')/(nsmooth*self.orbitdata.dTout)
        if track=='major':
            dMacc, time = self.time_smooth(iord,'dMtot',nsmooth=nsmooth,dosum=True)
            mdot_smooth = pynbody.array.SimArray(dMacc,'Msol')/(nsmooth*self.orbitdata.dTout)

        return mdot_smooth.in_units('Msol yr**-1'), time

    def smoothed_luminosity_history(self, iord, dt='10 Myr', er=0.1, fedd=False, track='major'):
        if fedd is False:
            mdot_smooth, time = self.smoothed_accretion_history(iord,dt,track)
            csq = phys_const['c'].in_units('cm s**-1')**2
            lum_smooth = mdot_smooth.in_units('g s**-1') * csq.in_units('erg g**-1') * er
        if fedd is True:
            tsmooth = pynbody.units.Unit(dt)
            nsmooth = int(tsmooth.ratio(self.orbitdata.dTout))
            lum_smooth = smoothdata(self[iord,'lum_er_'+str(er)]/ledd(self[iord,'mass',track]),nsteps=nsmooth,dosum=False)
            time = self[iord,'time',track][int(nsmooth/2)::nsmooth]
        return lum_smooth, time

