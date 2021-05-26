import os

import numpy as np
import pynbody
from ..util import param_reader

from bhtools.util import readcol


class BlackHoles(object):
	def __init__(self, simname, filename=None, paramfile=None):
		self.simname = simname
		if filename is None:
			self.filename = simname+'.BackHoles'
		else:
			self.filename = filename

		if paramfile is None:
			self.paramfile = simname+'.param'
		else:
			self.paramfile = paramfile

		if not os.path.exists(self.filename):
			raise RuntimeError("file", self.filename, "not found! Exiting...")

		#load in parameters from the param file - important for units!
		self.parameters = param_reader.ParamFile(simname)

		#read in data from simname.BlackHoles file
		iord, time, step, mass, x, y, z, vx, vy, vz, pot, Mdot, dM, dE, dt, dMaccum, dEaccum, a \
			= readcol.readcol(self.filename, twod=False, nanval=0.0)

		tsort = np.argsort(time)
		self._data = {
			'iord': iord[tsort].astype(np.int64),
			'time': pynbody.array.SimArray(time[tsort], self.paramfile.timeunit_st),
			'step': step[tsort],
			'mass': pynbody.array.SimArray(mass[tsort], self.paramfile.munit_st),
			'pos': pynbody.array.SimArray(np.concatenate([[x[tsort],y[tsort],z[tsort]]]).T, self.paramfile.dunit_st),
			'x': pynbody.array.SimArray(x[tsort], self.paramfile.dunit_st),
			'y': pynbody.array.SimArray(y[tsort], self.paramfile.dunit_st),
			'z': pynbody.array.SimArray(z[tsort], self.paramfile.dunit_st),
			'vel': pynbody.array.SimArray(np.concatenate([[vx,vy,vz]]).T, self.paramfile.velunit_st),
			'vx': pynbody.array.SimArray(vx[tsort], self.paramfile.velunit_st),
			'vy': pynbody.array.SimArray(vy[tsort], self.paramfile.velunit_st),
			'vz': pynbody.array.SimArray(vz[tsort], self.paramfile.velunit_st),
			'pot': pynbody.array.SimArray(pot[tsort], self.paramfile.potunit_st),
			'mdot': pynbody.array.SimArray(Mdot[tsort], self.paramfile.munit_st/self.paramfile.timeunit_st),
			'dM': pynbody.array.SimArray(mass[tsort], self.paramfile.munit_st),
			'dE': pynbody.array.SimArray(mass[tsort], self.paramfile.munit_st*self.paramfile.velunit_st**2),
			'dt': pynbody.array.SimArray(dt[tsort], self.paramfile.timeunit_st),
			'dMtot': pynbody.array.SimArray(dMaccum[tsort], self.paramfile.munit_st),
			'dEtot': pynbody.array.SimArray(dEaccum[tsort], self.paramfile.munit_st*self.paramfile.velunit_st**2),
			'a':a
		}

		self.bhiords, _id_slice = self._get_iord_slice_ind('iord')

		self.bhind = {}

		for i in range(len(self.bhiords)):
			self.bhind[self.bhiords[i]] = _id_slice[i]

	def __getitem__(self, item):
		if type(item)==str and item in self._data.keys() and ',' not in item:
			return self._data[item]
		if type(item)==str and ',' in item:
			select_num = item.split(',')[0]
			select_data = item.split(',')[1]

			if select_data not in self._data.keys():
				raise ValueError(select_data, " not found in BH data")
			else:
				return self._data[select_data][self.bhind[select_num]]



	def _get_iord_slice_ind(self):
		'''
		return unigue values of data[key] and the indices of the slices for those unique values
		'''
		ord_ = np.argsort(self.data['iord'])
		uvalues, ind = np.unique(self.data['iord'][ord_], return_index=True)
		slice_ = []
		for i in range(len(uvalues) - 1):
			ss = ord_[ind[i]:ind[i + 1]]
			sort_ = np.argsort(self.data['time'][ss])
			ss = ss[sort_]
			slice_.append(ss)
		#do last chunk
		ss = ord_[ind[i + 1]:]
		sort_ = np.argsort(self.data['time'][ss])
		ss = ss[sort_]
		slice_.append(ss)
		return uvalues, slice_

	def get_distance(self, ID1, ID2, comove=True):
		time1 = self[str(ID1)+','+'time']
		pos1 = self[str(ID1)+','+'pos']
		time2 = self[str(ID1) + ',' + 'time']
		pos2 = self[str(ID1) + ',' + 'pos']


		time2 = self.single_BH_data(ID2, 'time')
		x1 = self.single_BH_data(ID1, 'x')
		x2 = self.single_BH_data(ID2, 'x')
		y1 = self.single_BH_data(ID1, 'y')
		y2 = self.single_BH_data(ID2, 'y')
		z1 = self.single_BH_data(ID1, 'z')
		z2 = self.single_BH_data(ID2, 'z')
		scale = self.single_BH_data(ID1, 'scalefac')
		mint = np.max([time1.min(), time2.min()])
		maxt = np.minimum(time2.max(), time1.max())

		use1 = np.where((time1<=maxt)&(time1>=mint))[0]
		use2 = np.where((time2<=maxt)&(time2>=mint))[0]

		if len(use1) == 0 or len(use2) == 0:
			print("uh no no time match")
			return

		if len(use1) != len(use2):
			if len(use1)< len(use2):
				use1 = np.append(use1, use1[-1])
				if len(use1) != len(use2):
					print("SHIIIIIIT")
			else:
				print("SHIIIIIIIT oh no")

		xd = x1[use1]-x2[use2]
		yd = y1[use1]-y2[use2]
		zd = z1[use1]-z2[use2]

		bphys = boxsize*scale[use1]*1e3
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