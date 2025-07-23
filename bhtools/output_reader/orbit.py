import os

import numpy as np
import pynbody
from ..util import *

from bhtools.util import readcol

class BHOrbitData(object):
	def __init__(self, paramfile):
		self.paramfile = paramfile

		#load in parameters from the param file - important for units!
		self.parameters = param_reader.ParamFile(self.paramfile)

		#read in data from simname.BlackHoles file
		self._data = self._col_data()
		#convert to physical units

		self.bhiords, _id_slice = self._get_iord_slice_ind()

		self._bhind = {}

		for i in range(len(self.bhiords)):
			self._bhind[self.bhiords[i]] = _id_slice[i]

	def _col_data(self):
		return

	def __getitem__(self, item):
		'''
		user can get data under "key" for an individual BH with orbit_object[(iord,"key")]
		'''

		if type(item)!=str and type(item)!=tuple:
			raise ValueError("invalid arguments to __getitem__, use orbit_object[(iord,'key')]")

		if type(item)==str:
			if item in self._data.keys():
				return self._data[item]
			else:
				raise ValueError(item, "not found in BH data")

		if type(item)==tuple:
			select_id = item[0]
			select_data = item[1]

			if (type(select_id)!=np.int64 and type(select_id)!=int) or type(select_data)!=str:
				raise ValueError("invalid arguments to __getitem__, use orbit_object[(iord,'key')]")
			if select_data not in self._data.keys():
				raise ValueError(select_data, " not found in BH data")
			else:
				return self._data[select_data][self._bhind[select_id]]

	def _get_iord_slice_ind(self):
		'''
		return unigue iord values and the indices of the time-ordered data slices for each iord
		'''
		ord_ = np.argsort(self._data['iord'])
		uvalues, ind = np.unique(self._data['iord'][ord_], return_index=True)
		slice_ = []
		for i in range(len(uvalues) - 1):
			ss = ord_[ind[i]:ind[i + 1]]
			sort_ = np.argsort(self._data['time'][ss])
			ss = ss[sort_]
			#check for multiples in time and keep only the last one in the file
			utime, utind, cnt = np.unique(self._data['time'][ss], return_counts=True, return_index=True)
			double_ind = np.where(cnt>1)[0]
			to_cut = []
			for ii in double_ind:
				to_cut.extend(utind[ii:ii+cnt[ii]-1])
			ss = np.delete(ss,to_cut)
			slice_.append(ss)
		#do last chunk
		ss = ord_[ind[i + 1]:]
		sort_ = np.argsort(self._data['time'][ss])
		ss = ss[sort_]
		utime, utind, cnt = np.unique(self._data['time'][ss], return_counts=True, return_index=True)
		double_ind = np.where(cnt > 1)[0]
		to_cut = []
		for ii in double_ind:
			to_cut.extend(utind[ii:ii + cnt[ii] - 1])
		ss = np.delete(ss, to_cut)
		slice_.append(ss)
		return uvalues, slice_
	
	def _concatenate_iord_slices(self, bhiord, nodes_iord, nodes_time):
		#stithces together BHs at different times
		#meant to work with major progenitors
		if len(nodes_time)!=len(nodes_iord):
			raise RuntimeError("size of time and iord nodes must be the same")
		if len(nodes_time)==0: #no changes to main progenitor, so just give the original
			return self._bhind[bhiord]
		tmax = self._data['time'][self._bhind[bhiord]].max()
		tmin = nodes_time[0]
		slice_new = self._bhind[bhiord][(self._data['time'][self._bhind[bhiord]]>tmin)]
		for i in range(len(nodes_iord)):
			tmax = nodes_time[i]
			if len(nodes_time)>i+1:
				tmin = nodes_time[i+1]
			else:
				tmin = 0
			timearr = self._data['time'][self._bhind[nodes_iord[i]]]
			slice_new = np.append(self._bhind[nodes_iord[i]][(timearr>tmin)&(timearr<=tmax)], slice_new)
		return slice_new


	def time_smooth(self, iord, prop, nsmooth=10,ret_std=False, dosum=False):
		std = None
		if ret_std is True:
			data, std = smoothdata(self._data[prop][self._bhind[iord]], nsteps=nsmooth, ret_std=True, dosum=dosum)
		else:
			data = smoothdata(self._data[prop][self._bhind[iord]], nsteps=nsmooth, dosum=dosum)
		time = self._data['time'][self._bhind[iord]][int(nsmooth/2)::nsmooth]
		if len(time)>len(data):
			time = time[:-1]
		if std:
			return data, std, time
		else:
			return data, time
		
	#def get_all_distance(self, ID, time, comove=False, use_closest_time=True):
		

	def get_distance(self, ID1, ID2, comove=True):
		time1 = self[(ID1,'time')]
		time2 = self[(ID2,'time')]
		x1 = self[(ID1, 'x')]
		x2 = self[(ID2, 'x')]
		y1 = self[(ID1, 'y')]
		y2 = self[(ID2, 'y')]
		z1 = self[(ID1, 'z')]
		z2 = self[(ID2, 'z')]
		scale = self[(ID1, 'a')]

		use1 = np.where(np.in1d(time1,time2))[0]
		use2 = np.where(np.in1d(time2,time1))[0]

		if len(use1)!= len(use2)==0 or len(use1)==0 or len(use2)==0 or \
						len(np.where(time1[use1]!=time2[use2])[0])!=0:
			raise RuntimeError("bad time matching between BHs", ID1, ID2)

		xd = x1[use1]-x2[use2]
		yd = y1[use1]-y2[use2]
		zd = z1[use1]-z2[use2]

		#get the boxsize in physical units at each time in order to wrap distances
		boxsize_comove = self.parameters.get_boxsize()
		bphys = np.array([boxsize_comove.in_units('kpc', a=scale[ii]) for ii in use1])
		badx = np.where(np.abs(xd) > bphys/2)[0]

		xd[badx] = -1.0 * (xd[badx]/np.abs(xd[badx])) * \
							  np.abs(bphys[badx] - np.abs(xd[badx]))

		bady = np.where(np.abs(yd) > bphys/2)[0]
		yd[bady] = -1.0 * (yd[bady]/np.abs(yd[bady])) * \
							  np.abs(bphys[bady] - np.abs(yd[bady]))
		badz = np.where(np.abs(zd) > bphys/2)[0]
		zd[badz] = -1.0 * (zd[badz]/np.abs(zd[badz])) * \
							  np.abs(bphys[badz] - np.abs(zd[badz]))

		dist = np.sqrt(xd**2 + yd**2 + zd**2)

		if comove:
			dist /= scale[use1]
		return dist, time1[use1], scale[use1]**-1 -1
	
	def get_tform(self,sl):
		if sl is not None:
			sliords = sl['iord'].astype(np.int64)
			sliords[(sliords<0)] = 2*2147483648 + sliords[(sliords<0)]
			ord = np.argsort(sliords)
			bhind, = np.where(np.in1d(sliords[ord], self.bhiords))
			self.tform = sl['tform'][ord][bhind] * -1
			if self.tform.min() < 0: print("WARNING! Positive tforms were found for BHs!")
			gc.collect()
		else:
			cnt = 0
			self.tform = np.ones(len(self.bhiords)) * -1
			for id in self.bhiords:
				self.tform[cnt] = self.single_BH_data(id, 'time').min()
				cnt += 1

class BlackHoles(BHOrbitData):

	def _col_data(self):
		iord, time, step, mass, x, y, z, vx, vy, vz, pot, Mdot, dM, dE, dt, dMaccum, dEaccum, a \
			= readcol.readcol(self.filename, twod=False, nanval=0.0)

		tsort = np.argsort(time)
		data_dict = {
			'iord': iord[tsort].astype(np.int64),
			'time': pynbody.array.SimArray(time[tsort], self.parameters.timeunit_st),
			'step': step[tsort],
			'mass': pynbody.array.SimArray(mass[tsort], self.parameters.munit_st),
			'pos': pynbody.array.SimArray(np.concatenate([[x[tsort], y[tsort], z[tsort]]]).T, self.parameters.dunit_st),
			'x': pynbody.array.SimArray(x[tsort], self.parameters.dunit_st),
			'y': pynbody.array.SimArray(y[tsort], self.parameters.dunit_st),
			'z': pynbody.array.SimArray(z[tsort], self.parameters.dunit_st),
			'vel': pynbody.array.SimArray(np.concatenate([[vx[tsort], vy[tsort], vz[tsort]]]).T, self.parameters.velunit_st),
			'vx': pynbody.array.SimArray(vx[tsort], self.parameters.velunit_st),
			'vy': pynbody.array.SimArray(vy[tsort], self.parameters.velunit_st),
			'vz': pynbody.array.SimArray(vz[tsort], self.parameters.velunit_st),
			'pot': pynbody.array.SimArray(pot[tsort], self.parameters.potunit_st),
			'mdot': pynbody.array.SimArray(Mdot[tsort], self.parameters.munit_st / self.parameters.timeunit_st),
			'dM': pynbody.array.SimArray(mass[tsort], self.parameters.munit_st),
			'dE': pynbody.array.SimArray(mass[tsort], self.parameters.munit_st * self.parameters.velunit_st ** 2),
			'dt': pynbody.array.SimArray(dt[tsort], self.parameters.timeunit_st),
			'dMtot': pynbody.array.SimArray(dMaccum[tsort], self.parameters.munit_st),
			'dEtot': pynbody.array.SimArray(dEaccum[tsort], self.parameters.munit_st * self.parameters.velunit_st ** 2),
			'a': a
		}

		return data_dict

	def _phys_conv(self):
		scale = self._data['a']
		self._data['x'] = self._data['x'].in_units('kpc', a=scale)
		self._data['y'] = self._data['y'].in_units('kpc', a=scale)
		self._data['z'] = self._data['z'].in_units('kpc', a=scale)
		self._data['pos'] = self._data['pos'].in_units('kpc', a=np.concatenate([[scale],[scale],[scale]]).T)

		self._data['vx'] = self._data['vx'].in_units('km s**-1', a=scale)
		self._data['vy'] = self._data['vy'].in_units('km s**-1', a=scale)
		self._data['vz'] = self._data['vz'].in_units('km s**-1', a=scale)
		self._data['vel'] = self._data['vel'].in_units('km s**-1', a=np.concatenate([[scale],[scale],[scale]]).T)

		self._data['dMtot'] = self._data['dMtot'].in_units('Msol')
		self._data['dM'] = self._data['dM'].in_units('Msol')
		self._data['mass'] = self._data['mass'].in_units('Msol')

		self._data['mdot'] = self._data['mdot'].in_units('Msol yr**-1')

		self._data['dE'] = self._data['dE'].in_units('cm**2 s**-2 g', a=scale)
		self._data['dEtot'] = self._data['dEtot'].in_units('cm**2 s**-2 g', a=scale)
		self._data['pot'] = self._data['pot'].in_units('km**2 s**-2', a=scale)

		self._data['dt'] = self._data['dt'].in_units('Gyr')
		self._data['time'] = self._data['time'].in_units('Gyr')

	def __init__(self, simname, filename=None, paramfile=None):
		if paramfile is None:
			self.paramfile = simname+'.param'
		else:
			self.paramfile = paramfile
		if filename is None:
			self.filename = simname+'.BlackHoles'
		else:
			self.filename = filename

		if not os.path.exists(self.filename):
			raise RuntimeError("file", self.filename, "not found! Exiting...")

		super().__init__(self.paramfile)

		self.dTout = self._get_output_cadence()
		self._phys_conv()

	def calc_lum(self, er=0.1):
		self._data['lum_er_'+str(er)] = self._data['mdot'].in_units('g s**-1')*phys_const['c'].in_units('cm s**-1')**2 *er
		

	def _get_output_cadence(self):
		t0 = cosmology.getTime(0,self.parameters.h, self.parameters.omegaM, self.parameters.omegaL,unit='Gyr')
		dt_big = t0 / int(self.parameters.params['nSteps'])
		dt_out = dt_big / 2**int(self.parameters.params['iBHSinkOutRung'])
		return pynbody.units.Unit(str(dt_out)+' Gyr')

	def smoothed_accretion_history(self, iord, dt='10 Myr'):
		tsmooth = pynbody.units.Unit(dt)
		nsmooth = int(tsmooth.ratio(self.dTout))
		dMacc, time = self.time_smooth(iord, 'dMtot', nsmooth=nsmooth, dosum=True)
		mdot_smooth = pynbody.array.SimArray(dMacc,'Msol')/(nsmooth*self.dTout)
		return mdot_smooth.in_units('Msol yr**-1'), time

	def smoothed_luminosity_history(self, iord, dt='10 Myr', er=0.1):
		mdot_smooth, time = self.smoothed_accretion_history(iord,dt)
		csq = phys_const['c'].in_units('cm s**-1')**2
		lum_smooth = mdot_smooth.in_units('g s**-1') * csq.in_units('erg g**-1') * er
		return lum_smooth, time

class ConvertOldOrbit(BHOrbitData):
	def __init__(self, old_pickle_file, paramfile):
		self.paramfile = paramfile
		self.old_pickle_file = old_pickle_file
		super().__init__(paramfile)

	def _col_data(self):
		import pickle
		data_dict = {}
		f = open(self.old_pickle_file, 'rb')
		try:
			old_orbit = pickle.load(f)
		except:
			f.close()
			f = open(self.old_pickle_file,'rb')
			old_orbit = pickle.load(f,encoding='latin1')
		f.close()
		print("Successfully read in old pickle file", self.old_pickle_file)
		print("Gathering data", old_orbit.data.keys())
		for key in old_orbit.data.keys():
			data_dict[key] = old_orbit.data[key]
		return data_dict


