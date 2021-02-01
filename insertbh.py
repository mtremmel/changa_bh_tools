import pynbody
import numpy as np
import struct
import os

def insert_bhs(sim, step, halos, filename, bhmass=1e5, part_center=32):
	import tangos as db
	bhdata = create_central_bh(sim, step, halos, part_center=part_center, bhmass=bhmass)
	simfolder = os.getenv('TANGOS_SIMULATION_FOLDER')
	snapfile = simfolder+'/'+db.get_timestep(sim+'/%'+str(step)).path
	s = pynbody.load(snapfile)
	create_bh_tipsy_file(s, len(halos), filename, bhdata=bhdata)

def insert_tipsy_array_file(snap, array_name, values, binary=True, filename=None):
	if type(snap) == pynbody.snapshot.tipsy.TipsySnap:
		s = snap
	elif type(snap) == str:
		s = pynbody.load(snap)

	if array_name in s._basic_loadable_keys[pynbody.family.star]:
		raise ValueError("array name provided is a basic property. use insert_tipsy instead")
	try:
		ar = s[array_name]
	except:
		raise ValueError("array name given cannot be found in the simulation!")

	ar_new = np.append(np.asarray(ar, dtype=ar.dtype), values)

	if not filename:
		filename = s._filename+'.'+array_name

	if binary:
		f = open(filename, 'wb')
		if s._byteswap:
			f.write(struct.pack('>i', len(s)+1))
			ar_new.byteswap().tofile(f)
		else:
			f.write(struct.pack('i', len(s)+1))
			ar_new.tofile(f)
		f.close()

	else:
		f = open(filename,'w')
		f.write((str(len(s)+1)+'\n').encode('utf-8'))
		if issubclass(ar_new.dtype, np.integer):
			fmt='%d'
		else:
			fmt='%e'
		np.savetxt(f, ar_new, fmt=fmt)
		f.close()


def create_bh_tipsy_file(snap, nbhs, filename, bhdata=None):
	# read in original snapshot
	if type(snap) == pynbody.snapshot.tipsy.TipsySnap:
		s = snap
	elif type(snap) == str:
		s = pynbody.load(snap)
	ndm = len(s.dm)
	ng = len(s.g)
	ns = len(s.s)

	#check provided data
	if bhdata:
		if not hasattr(bhdata,'keys'):
			raise ValueError("Error: bhdata provided needs to be in a dictionary-like format")
		for key in bhdata.keys():
			if len(bhdata[key]) != nbhs:
				raise ValueError("Error: bhdata provided does not have the correct number of values for", key)
			if not hasattr(bhdata[key],'units')
				raise ValueError("Error: bhdata provided must be SimArray objects with units")
			if bhdata[key].units == pynbody.units.NoUnit():
				raise ValueError("Error: bhdata for ", key, " does not have units")
			#prime the original data by reading in all of the provided bh data
			if key in s._basic_loadable_keys() or key in s.loadable_keys():
				try:
					s[key]
				except:
					print("warning:", key, "Not able to be loaded in original snapshot")
					continue

	#create a new snapshot from scratch
	new_snap = pynbody.new(dm=ndm, gas=ng, star=ns+nbhs, order='gas,dm,star')


	for key in s.dm.keys():
		new_snap.dm[key] = s.dm[key].in_units(s.infer_original_units(s.dm[key].units))
	for key in s.g.keys():
		new_snap.g[key] = s.g[key].in_units(s.infer_original_units(s.g[key].units))
	for key in s.s.keys():
		if bhdata and key in bhdata.keys():
			bhvals = bhdata[key].in_units(s.infer_original_units(s.s[key].units)) #make sure to match the original simulation units!
		else:
			print("data for ", key, " was not provided, using 0.0")
			bhvals = np.zeros(nbhs)
		new_snap.s[key] = pynbody.array.SimArray(np.append(np.asarray(
			s.s[key].in_units(s.infer_original_units(s.s[key].units))), np.asarray(bhvals)), s.s[key].units)

	new_snap._byteswap = s._byteswap
	new_snap.properties = s.properties
	new_snap.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=filename)


def create_central_bh(sim, step, halo_numbers, part_center=32, bhmass=1e5):
	"""
	:param sim: name of simulation (string)
	:param step: target step number (string or integer)
	:param halo_number: list of target halo numbers at the given step (list of integers)
	:param part_center: number of particles to use in velocity calculation
	:param bhmass: the mass of the bhs to place (Msol)
	:return:
	"""

	import tangos as db
	ts = db.get_timestep(sim+'/%'+str(step))
	simfolder = os.getenv('TANGOS_SIMULATION_FOLDER')
	snapfile = simfolder + '/' + ts.path
	print('getting data from ', snapfile)
	s = pynbody.load(snapfile)
	h = s.halos(dosort=True)

	bhdata = {}
	for key in s.loadable_keys():
		bhdata[key] = []

	for hh in halo_numbers:
		print("getting BH data for halo", hh)
		target = db.get_halo(sim+'/%'+str(step)+'/'+str(hh))
		position = pynbody.array.SimArray(target['shrink_center'], 'kpc')
		bhdata['pos'].append(position.in_units(s.infer_original_units('kpc')))

		ht = h.load_copy(hh)

		ht['pos'] -= position.in_units(s.infer_original_units('kpc'))
		ht.wrap()


		r_sdm = ht[(pynbody.filt.FamilyFilter(pynbody.family.star)|pynbody.filt.FamilyFilter(pynbody.family.dark))]['r']
		phi_sdm = ht[(pynbody.filt.FamilyFilter(pynbody.family.star)|pynbody.filt.FamilyFilter(pynbody.family.dark))]['phi']
		rmax = r_sdm[np.argsort(r_sdm)][part_center-1]
		cen = ht[(pynbody.filt.FamilyFilter(pynbody.family.star)|pynbody.filt.FamilyFilter(pynbody.family.dark))
		         & pynbody.filt.Sphere(rmax)]

		vcen = (cen['vel'].transpose() * cen['mass']).sum(axis=1) / \
	       cen['mass'].sum()

		bhdata['vel'].append(vcen)

		pot = phi_sdm[np.argmin(r_sdm)]
		bhdata['phi'].append(pot)

		bhdata['tform'].append(-1)
		bhdata['metals'].append(0)
		bhdata['eps'].append(ht.s['eps'].min())
		bhdata['mass'].append(bhmass)

	for key in bhdata.keys():
		if key == 'mass':
			unit = 'Msol'
		else:
			unit = ht.s[key].units
		bhdata[key] = pynbody.array.SimArray(bhdata.keys, unit)

	return bhdata









		
		



