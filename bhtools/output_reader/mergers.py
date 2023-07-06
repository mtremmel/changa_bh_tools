import numpy as np
from .. import util
import os
import pynbody

def get_mergers_by_id(bhiord, mdata, time, dmin, dtmin):
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
	:param mdata: merger data
	:return:
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
	def __init__(self, simname, paramfile):
		self.simname = simname
		self.parameters = util.param_reader.ParamFile(self.paramfile)
		self.db_mergers = {}
		if paramfile is None:
			self.paramfile = simname+'.param'
		else:
			self.paramfile = paramfile

		self.mergerfile = simname + '.mergers'
		print("reading .mergers file...")
		time, step, ID, IDeat, ratio, kick = util.readcol.readcol(self.mergerfile, twod=False)
		print("checking for bad IDs...")
		bad = np.where(ID < 0)[0]
		if len(bad) > 0:
			ID[bad] = 2 * 2147483648 + ID[bad]
		bad2 = np.where(IDeat < 0)[0]
		if len(bad2) > 0:
			IDeat[bad2] = 2 * 2147483648 + IDeat[bad2]

		uIDeat, indices = np.unique(IDeat, return_index=True)

		self.rawdat = {'time': time, 'ID1': ID, 'ID2': IDeat, 'ratio': ratio, 'kick': kick, 'step': step}
		util.cutdict(self.rawdat, indices)
		ordr = np.argsort(self.rawdat['ID2'])
		util.cutdict(self.rawdat, ordr)

		z = util.cosmology.getRedshift(pynbody.array.SimArray(self.rawdat['Time'],'Gyr'),
		                               self.parameters.h, self.parameters.omegaM, self.parameters.omegaL)

		self.rawdat['redshift'] = z

		uIDeat, cnt = np.unique(self.rawdat['ID2'], return_counts=True)
		if len(np.where(cnt > 1)[0]) > 0:
			raise RuntimeWarning("Same Black Hole Marked as Eaten TWICE!")

