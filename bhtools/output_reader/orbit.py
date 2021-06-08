import os

import numpy as np
import pynbody
from ..util import param_reader

from bhtools.util import readcol


class BlackHoles(object):

	def _col_data(self, filename):
		iord, time, step, mass, x, y, z, vx, vy, vz, pot, Mdot, dM, dE, dt, dMaccum, dEaccum, a \
			= readcol.readcol(filename, twod=False, nanval=0.0)

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
			'vel': pynbody.array.SimArray(np.concatenate([[vx, vy, vz]]).T, self.parameters.velunit_st),
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

	def __init__(self, simname, filename=None, paramfile=None):
		self.simname = simname
		if filename is None:
			self.filename = simname+'.BlackHoles'
		else:
			self.filename = filename

		if paramfile is None:
			self.paramfile = simname+'.param'
		else:
			self.paramfile = paramfile

		if not os.path.exists(self.filename):
			raise RuntimeError("file", self.filename, "not found! Exiting...")

		#load in parameters from the param file - important for units!
		self.parameters = param_reader.ParamFile(self.paramfile)
		self.parameters._get_basic_units()

		#read in data from simname.BlackHoles file
		self._data = self._col_data(self.filename)

		self.bhiords, _id_slice = self._get_iord_slice_ind()

		self._bhind = {}

		for i in range(len(self.bhiords)):
			self._bhind[self.bhiords[i]] = _id_slice[i]

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

			if type(select_id)!=int or type(select_data)!=str:
				raise ValueError("invalid arguments to __getitem__, use orbit_object[(iord,'key')]")
			if select_data not in self._data.keys():
				raise ValueError(select_data, " not found in BH data")
			else:
				return self._data[select_data][self._bhind[select_id]]

	def _get_iord_slice_ind(self):
		'''
		return unigue values of data[key] and the indices of the slices for those unique values
		'''
		ord_ = np.argsort(self._data['iord'])
		uvalues, ind = np.unique(self._data['iord'][ord_], return_index=True)
		slice_ = []
		for i in range(len(uvalues) - 1):
			ss = ord_[ind[i]:ind[i + 1]]
			sort_ = np.argsort(self._data['time'][ss])
			ss = ss[sort_]
			slice_.append(ss)
		#do last chunk
		ss = ord_[ind[i + 1]:]
		sort_ = np.argsort(self._data['time'][ss])
		ss = ss[sort_]
		slice_.append(ss)
		return uvalues, slice_

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