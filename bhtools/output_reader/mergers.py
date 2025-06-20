import numpy as np
from .. import util
import os
import glob
import pynbody

def get_mergers_by_id(bhiord, mdata, time, dmin, dtmin):
	'''
		Extract information on mergers which involve a specific BH ID number.
		:param bhiord: target BH id number
		:param mdata: merger data
		:param time: maximum time to consider
		:param dmin: minimum initial distance for "true" mergers
		:param dtmin: minimum time since formation for "true" mergers
		:return: IDs of "true" merger BHs, Merger times, Formation times of each BH (1 and 2),
		initial distance between BHs, exhaustive list of IDs for ALL mergers in the simulation
		'''
	# return mdata['IDeat'][(mdata['ID']==bhiord)], mdata['step'][(mdata['ID']==bhiord)]
	match = np.where((mdata['ID1'] == bhiord) & (mdata['merge_mass_1'] >= 1e6) & (mdata['merge_mass_2'] >= 1e6) & (
				np.minimum(mdata['tform1'], mdata['tform2']) > 0)&(mdata['time']<time)&
	                 (mdata['time']-np.maximum(mdata['tform1'], mdata['tform2'])>dtmin)&
	                 (mdata['init_dist']>dmin))[0]
	match_all = np.where((mdata['ID1'] == bhiord)&(mdata['time']<time))[0]
	# strict = BHs formed at initial separations greater than a kpc
	return mdata['ID2'][match], mdata['time'][match], mdata['tform1'][match], mdata['tform2'][match], \
	       mdata['init_dist'][match], mdata['ID2'][match_all]

def get_all_mergers(bhiord, mdata, time, dmin, dtmin):
	'''
	Extract an exhaustive list of mergers along all branches of a target BH's merger tree
	:param bhiord: target BH id number
	:param mdata: merger data
	:param time: maximum time to consider
	:param dmin: minimum initial distance for "true" mergers
	:param dtmin: minimum time since formation for "true" mergers
	:return: IDs of "true" merger BHs, Merger times, Formation times of each BH (1 and 2),
	initial distance between BHs, exhaustive list of IDs for ALL mergers in the simulation
	'''
	bhlist = list([bhiord])
	bhlist_new = list([])
	id_list_all = list([])
	id_list = list([])
	id_list_strict = list([])
	time_list = list([])
	dist_list = list([])
	tform1_list = list([])
	tform2_list = list([])
	while len(bhlist) > 0:
		for i in range(len(bhlist)):
			bhlist_part, time_part, tform1_part, tform2_part, dist_part, id_all_part = get_mergers_by_id(bhlist[i],
			                                                                                             mdata, time,
			                                                                                             dmin, dtmin)
			bhlist_new.extend(bhlist_part)
			id_list.extend(bhlist_part)
			time_list.extend(time_part)
			tform1_list.extend(tform1_part)
			tform2_list.extend(tform2_part)
			id_list_all.extend(id_all_part)
			dist_list.extend(dist_part)
		bhlist = bhlist_new
		bhlist_new = list([])
	time_list = pynbody.array.SimArray(time_list, 'Gyr')
	tform1_list = pynbody.array.SimArray(tform1_list, 'Gyr')
	tform2_list = pynbody.array.SimArray(tform2_list, 'Gyr')
	dist_list = pynbody.array.SimArray(dist_list, 'kpc')

	return np.array(id_list), time_list, tform1_list, tform2_list, dist_list, np.array(id_list_all)

def collect_all_bh_mergers(tot_bhids, time, mdata, dmin, dtmin):
	'''
	:param tot_bhids: the total list of BH ids you want to collect mergers for
	:param time: maximum time of mergers to consider
	:param mdata: merger data object
	:param dmin: the minimum initial separation of BHs to be "true" mergers
	:param dtmin: the minimum time between formation and merger "true" mergers
	:return: total number of "true" mergers, "true" merger times, "true" merger BH IDs,
	All merger BH IDs, time of last "true" merger, final BH ID (same as input), formation time
	of "True" mergers (1 and 2), and the total number of All mergers.
	'''
	#tot_bhids = np.append(bhid_cen, bhid_any)
	#tot_bhids = np.unique(tot_bhids)
	nmerge = np.ones(len(tot_bhids)) * -1
	nmerge_all = np.ones(len(tot_bhids)) * -1
	tmerge = list([])
	iord_merge = list([])
	iord_merge_all = list([])
	hid_bh = tot_bhids
	tlast = np.ones(len(tot_bhids)) * -1
	tform1 = list([])
	tform2 = list([])
	init_dist = list([])

	for i in range(len(tot_bhids)):
		frac = i / len(tot_bhids)
		if frac % 0.1 < 1 / len(tot_bhids): print(frac)
		prog_id, tm, tf1, tf2, dd, prog_all = get_all_mergers(tot_bhids[i], mdata, time, dmin, dtmin)
		nmerge[i] = len(prog_id)
		nmerge_all[i] = len(prog_all)
		tmerge.append(tm)
		iord_merge.append(prog_id)
		iord_merge_all.append(prog_all)
		tform1.append(tf1)
		tform2.append(tf2)
		init_dist.append(dd)
		if len(prog_id) > 0:
			tlast[i] = tm.max()

	return nmerge, tmerge, iord_merge, iord_merge_all, tlast, hid_bh, tform1, tform2, nmerge_all



class BHMergers(object):
	def __init__(self, simname, paramfile=None, use_existing_object=None):
		'''
		:param simname: name of simulation (based on output file names)
		:param paramfile: name of param file (default is simname.param)
		:param use_existing_object: initialize a new object from an old one
		'''	
		self.simname = simname
		self.db_mergers = {}
		if paramfile is None:
			self.paramfile = simname+'.param'
		else:
			self.paramfile = paramfile
		self.parameters = util.param_reader.ParamFile(self.paramfile)
		if use_existing_object:
			self._initialize_from_existing(use_existing_object)
		else:
			self.mergerfile = simname + '.BHmergers'
			print("reading .mergers file...")
			ID, IDeat, Mass1, Mass2, ratio, kick, time, scale = util.readcol.readcol(self.mergerfile, twod=False)
			print("checking for bad IDs...")
			bad = np.where(ID < 0)[0]
			if len(bad) > 0:
				ID[bad] = 2 * 2147483648 + ID[bad]
			bad2 = np.where(IDeat < 0)[0]
			if len(bad2) > 0:
				IDeat[bad2] = 2 * 2147483648 + IDeat[bad2]
			
			testsnap = glob.glob('simname.000???')[0]f = pynbody.load(testsnap)
			tunits = f.infer_original_units('Gyr')
			munits = f.infer_original_units('Msol')
			gyr_ratio = pynbody.units.Gyr.ratio(tunits)
			msol_ratio = pynbody.units.Msol.ratio(munits)

			uIDeat, indices = np.unique(IDeat, return_index=True)

			self.rawdat = {'time': time/gyr_ratio, 'ID1': ID, 'ID2': IDeat, 'ratio': ratio, 'kick': kick, 'scale': scale,
			               'redshift': scale**-1 -1, 'merge_mass_1': Mass1/msol_ratio, 'merge_mass_2':Mass2/msol_ratio}
			util.cutdict(self.rawdat, indices)
			ordr = np.argsort(self.rawdat['ID2'])
			util.cutdict(self.rawdat, ordr)

			uIDeat, cnt = np.unique(self.rawdat['ID2'], return_counts=True)
			if len(np.where(cnt > 1)[0]) > 0:
				raise RuntimeWarning("Same Black Hole Marked as Eaten TWICE!")

	def __getitem__(self,item):
		return self.rawdat[item]

	def keys(self):
		return self.rawdat.keys()

	def _initialize_from_existing(self,merger_file):
		print("creating raw data from existing merger file")
		print("exiting keys: ", merger_file.keys())
		self.rawdat = {}
		for key in merger_file.keys():
			self.rawdat[key] = merger_file[key]

	def _get_redshifts(self):
		z = util.cosmology.getRedshift(pynbody.array.SimArray(self.rawdat['time'], 'Gyr'),
		                               self.parameters.h, self.parameters.omegaM, self.parameters.omegaL)
		self.rawdat['redshift'] = z
	
	def get_tform(self,bhorbit):
		self.rawdat['tform1'] = np.ones(len(self.rawdat['ID1']))*-1
		self.rawdat['tform2'] = np.ones(len(self.rawdat['ID2']))*-1

		for i in range(len(self.rawdat['ID2'])):
			time1 = bhorbit[self.rawdat['ID1'][i], 'time']
			time2 = bhorbit[self.rawdat['ID2'][i], 'time']
			if len(time1)>0:
				self.rawdat['tform1'][i] = bhorbit.tform[(bhorbit.bhiords==self.rawdat['ID1'][i])][0]
			if len(time2)>0:
				self.rawdat['tform2'][i] = bhorbit.tform[(bhorbit.bhiords==self.rawdat['ID2'][i])][0]

	def get_initial_distance(self,bhorbit):
		self.rawdat['init_dist'] = np.ones(len(self.rawdat['ID1']))*-1
		for i in range(len(self.rawdat['ID1'])):
			try:
				d12, t12, z12 = bhorbit.get_distance(self.rawdat['ID1'][i], self.rawdat['ID2'][i], comove=False)
				self.rawdat['init_dist'][i] = d12[np.argmin(t12)]
			except:
				continue

	def get_final_mdot(self, bhorbit):
		self.rawdat['mdot_final_1'] = np.ones(len(self.rawdat['ID1']))*-1
		self.rawdat['mdot_final_2'] = np.ones(len(self.rawdat['ID2'])) * -1
		for i in range(len(self.rawdat['ID2'])):
			tmerger = self.rawdat['time'][i]
			tlast_orbit = bhorbit[self.rawdat['ID2'][i],'time'].max()
			if tlast_orbit-tmerger > 0.005:
				print("Black Hole", self.rawdat['ID2'][i],
				            "has a mismatch between merger data and orbit data times")
				continue
			self.rawdat['mdot_final_1'][i] = bhorbit[self.rawdat['ID2'][i],'mdot'][-1]
			time_bh_1 = bhorbit[self.rawdat['ID1'][i],'time']
			self.rawdat['mdot_final_2'][i] = bhorbit[self.rawdat['ID1'][i], 'mdot'][np.argmin(np.abs(time_bh_1-tmerger))]

	def get_dual_frac(self, bhorbit,minL=1e43,maxD=10,comove=True, gather_array=False, timestep=None, er=0.1):

		tstr = 't_D' + str(maxD)
		fstr = 'frdual_L' + str(minL) + '_D' + str(maxD)
		if comove:
			tstr = tstr + 'c'
			fstr = fstr + 'c'

		self.rawdat[fstr] = np.ones(len(self.rawdat['ID1'])) * -1
		self.rawdat[tstr] = np.ones(len(self.rawdat['ID1'])) * -1

		for i in range(len(self.rawdat['ID1'])):
			if self.rawdat['ID2'][i] not in  bhorbit['iord'] or self.rawdat['ID1'][i] not in bhorbit['iord']:
				continue

			time1 = bhorbit[self.rawdat['ID1'][i], 'time']
			mdot1 = bhorbit[self.rawdat['ID1'][i], 'mdot'].in_units('g s**-2')
			time2 = bhorbit[self.rawdat['ID2'][i], 'time']
			mdot2 = bhorbit[self.rawdat['ID2'][i], 'mdot'].in_units('g s**-2')

			if time2.max()-self.rawdat['time'][i]>0.005:
				print("BHs have a mismatch in orbit data! Fake merger?",
				      self.rawdat['ID1'][i], self.rawdat['ID2'][i])
				continue
			if time2.max()<time1.min():
				print("BH merger and formation of one or both coincide",
				      self.rawdat['ID1'][i], self.rawdat['ID2'][i])
				continue
			try:
				distance, time, scale = bhorbit.get_distance(self.rawdat['ID1'][i], self.rawdat['ID2'][i], comove=comove)
			except:
				print("Failed to calculate distances... ignoring", self.rawdat['ID1'][i], self.rawdat['ID2'][i])
				continue
			if len(time)<2:
				print("weird! zero times!", distance, self.rawdat['ID1'][i], self.rawdat['ID2'][i])
				continue
			close = np.where(distance < maxD)[0]

			use1 = np.where(np.in1d(time1, time2))[0]
			use2 = np.where(np.in1d(time2, time1))[0]
			use3 = np.where(np.in1d(time,time1[use1]))[0]
			if len(use3)!=len(time) or len(np.where(time[use3]!=time1[use1])[0])!=0:
				print("Warning! Issue with mapping for luminosities! Skipping...",
				      self.rawdat['ID1'][i], self.rawdat['ID2'][i])
				continue
			lum1 = util.phys_const['c']**2 * mdot1[use1] * er
			lum2 = util.phys_const['c'] ** 2 * mdot2[use2] * er

			close_and_bright = np.where((distance<maxD)&(lum1>minL)&(lum2>minL))[0]

			if timestep is None: #estimate timesteps between outputs
				dt = time[1:]-time[:-1]
				dt = np.append(dt[0],dt)
				time_close = np.sum(dt[close])
				time_bright = np.sum(dt[close_and_bright])
			else: #use provided constant output timestep
				time_close = timestep*len(close)
				time_bright = timestep*len(close_and_bright)
			self.rawdat[tstr][i] = time_close
			self.rawdat[fstr][i] = time_bright/time_close






