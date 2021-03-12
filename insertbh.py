import pynbody
import numpy as np
import struct
import os, sys, glob

def insert_bhs(sim, step, halos, filename, bhmass=1e5, part_center=32):
	import tangos as db
	bhdata, delete_iords, new_mass, new_mass_iords = create_central_bh(sim, step, halos, part_center=part_center, bhmass=bhmass)
	delete_iords = np.concatenate(delete_iords)
	new_mass = np.concatenate(new_mass)
	new_mass_iords = np.concatenate(new_mass_iords)
	print("gas particles set for deletion:", delete_iords)
	print("gas particles set for new masses:", new_mass_iords)
	print("new masses for gas:", new_mass)
	simfolder = os.getenv('TANGOS_SIMULATION_FOLDER')
	snapfile = simfolder+'/'+db.get_timestep(sim+'/%'+str(step)).path
	s = pynbody.load(snapfile)
	slfile_list = glob.glob(os.path.join(simfolder+'/'+sim,'*.starlog'))
	if (len(slfile_list)):
		slfile = slfile_list[0]
	else:
		raise IOError("starlog file not found!")
	print("reading starlog!")
	sl = pynbody.snapshot.tipsy.StarLog(slfile)
	print("creating tipsy file...")
	create_bh_tipsy_file(s, len(halos), filename+'.'+str(step),
	                     bhdata=bhdata, delete_iords=delete_iords,
	                     newmass_iords=new_mass_iords, newmasses=new_mass)
	create_bh_starlog(s, sl, bhdata, filename+'.starlog')
	return

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


def create_bh_tipsy_file(snap, nbhs, filename, bhdata=None, delete_iords=None, newmass_iords=None, newmasses=None):
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
			if not hasattr(bhdata[key],'units'):
				raise ValueError("Error: bhdata provided must be SimArray objects with units")

	#load in array files by hand to ensure pynbody won't try to use the starlog
	for key in s.loadable_keys():
		if key not in s._basic_loadable_keys['star'].union(s._basic_loadable_keys['gas'], s._basic_loadable_keys['dm']):
			s._load_array(key, filename=s._filename+'.'+key)

	# remove deleted gas particles flagged to 'make' the black hole
	if delete_iords is not None:
		ng -= len(delete_iords) #make sure we create a tipsy file with the right number of particles!
		s.g = s.g[(np.in1d(s.g['iord'], delete_iords) == False)]

	# assign new masses to partially deleted gas mass
	#  this mass is assumed to have gone into 'making' the black hole
	if newmasses is not None:
		if not newmass_iords:
			raise ValueError("newmass_iords must come with list of new masses")
		if len(newmasses) != len(newmass_iords):
			raise ValueError("list newmass_iords must have same length as newmasses!")
		for i in range(len(newmass_iords)):
			s.g[(s.g['iord'] == newmass_iords[i])]['mass'] = newmasses[i]

	# create a new snapshot from scratch
	new_snap = pynbody.new(dm=ndm, gas=ng, star=ns+nbhs, order='gas,dm,star')

	for key in s.dm.loadable_keys():
		print("loading dark matter sim data for", key)
		if s.dm[key].units != pynbody.units.NoUnit():
			new_snap.dm[key] = s.dm[key].in_units(s.infer_original_units(s.dm[key].units), a=s.properties['a'])
		else:
			new_snap.dm[key] = s.dm[key]
	for key in s.g.loadable_keys():
		print("loading gas sim data for", key)
		if s.g[key].units != pynbody.units.NoUnit():
			new_snap.g[key] = s.g[key].in_units(s.infer_original_units(s.g[key].units), a=s.properties['a'])
		else:
			new_snap.g[key] = s.g[key]
	for key in s.s.loadable_keys():
		print("loading and adding new star data for", key)
		if bhdata and key in bhdata.keys():
			# make sure to match the original simulation units!
			if bhdata[key].units != pynbody.units.NoUnit():
				bhvals = bhdata[key].in_units(s.infer_original_units(s.s[key].units), a=s.properties['a'])
			else:
				bhvals = bhdata[key]
		else:
			print("data for ", key, " was not provided, using 0.0")
			bhvals = np.zeros(nbhs)
		if len(np.shape(s.s[key])) == 2:
			ashape = (ns + nbhs, 3)
		else:
			ashape = (ns + nbhs,)
		if s.s[key].units != pynbody.units.NoUnit():
			new_snap.s[key] = pynbody.array.SimArray(
				np.append(np.asarray(s.s[key].in_units(s.infer_original_units(s.s[key].units),a=s.properties['a'])),
				np.asarray(bhvals)).reshape(ashape),
				s.infer_original_units(s.s[key].units))
		else:
			new_snap.s[key] = pynbody.array.SimArray(
				np.append(np.asarray(s.s[key]),np.asarray(bhvals)).reshape(ashape))

	new_snap._byteswap = s._byteswap
	new_snap.properties = s.properties
	print("writing data...")
	new_snap.write(fmt=pynbody.snapshot.tipsy.TipsySnap, filename=filename, binary_aux_arrays=True)


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
	iord_max = s['iord'].max()

	bhdata = {}
	units = {}
	for key in s.s.loadable_keys():
		bhdata[key] = []
		units[key] = ''

	bhcount = 1
	delete_iords = []
	new_mass_iords = []
	new_mass = []
	for hh in halo_numbers:
		print("getting BH data for halo", hh)
		target = db.get_halo(sim+'/%'+str(step)+'/'+str(hh))
		position = pynbody.array.SimArray(target['shrink_center'], 'kpc')
		bhdata['pos'].append(position.in_units(s.infer_original_units('kpc'), a=s.properties['a']))

		ht = h.load_copy(hh)
		ht['pos'] -= position.in_units(s.infer_original_units('kpc'), a = s.properties['a'])
		ht.wrap()

		#select particles to delete or remove mass from to create BH
		delete_part, new_mass_part, new_mass_iord_part = select_gas_particles(ht, bhmass, 2*ht.s['eps'].min()/ht.properties['a'])
		delete_iords.append(delete_part)
		new_mass_iords.append(new_mass_iord_part)
		new_mass.append(new_mass_part)

		r_sdm = ht[(pynbody.filt.FamilyFilter(pynbody.family.star)|pynbody.filt.FamilyFilter(pynbody.family.dm))]['r']
		phi_sdm = ht[(pynbody.filt.FamilyFilter(pynbody.family.star)|pynbody.filt.FamilyFilter(pynbody.family.dm))]['phi']
		rmax = r_sdm[np.argsort(r_sdm)][part_center-1]
		cen = ht[(pynbody.filt.FamilyFilter(pynbody.family.star)|pynbody.filt.FamilyFilter(pynbody.family.dm))
		         & pynbody.filt.Sphere(rmax)]

		vcen = (cen['vel'].transpose() * cen['mass']).sum(axis=1) / \
	       cen['mass'].sum()

		bhdata['vel'].append(vcen)

		pot = phi_sdm[np.argmin(r_sdm)]
		bhdata['phi'].append(pot)

		bhdata['tform'].append(-1)
		bhdata['rung'].append(6) #guess at a reasonable BH rung just in case
		bhdata['eps'].append(ht.s['eps'].min())
		bhdata['mass'].append(bhmass)
		if 'massform' in bhdata.keys():
			bhdata['massform'].append(bhmass)
		bhdata['iord'].append(iord_max+bhcount)
		bhcount+=1
		for key in bhdata.keys(): #fill in the rest of the available auxillary data with zeros
			if key not in ['pos','vel','mass','eps','tform','rung','phi', 'iord', 'massform']:
				bhdata[key].append(0)

	units['mass'] = 'Msol' #the BH masses are always expected to be given in solar masses just to keep things user friendly
	for key in bhdata.keys():
		print("initializing units for", key)
		if key != 'mass' and key!='massform':
			try:
				units[key] = ht.s[key].units
			except:
				units[key] = ht[key].units
		if key == 'massform': #try to avoid massform which shows up often as an array file and in the starlog
			units[key] = 'Msol'

	for key in bhdata.keys():
		bhdata[key] = pynbody.array.SimArray(bhdata[key], units[key])

	return bhdata, delete_iords, new_mass, new_mass_iords


def get_starlog_meta(sl):
	with open(sl._logfile, 'r') as g:
		print("reading in starlog meta data...")
		read_metadata = False
		structure_names = []
		structure_formats = []
		for line in g:
			if line.startswith('# end starlog data'):
				read_metadata = False
			if read_metadata:
				meta_name, meta_type = line.strip('#').split()
				meta_name = sl._infer_name_from_tipsy_log(meta_name)
				structure_names.append(meta_name)
				structure_formats.append(meta_type)
			if line.startswith('# starlog data:'):
				read_metadata = True
	file_structure = np.dtype({'names': structure_names,
	                           'formats': structure_formats})
	return file_structure

def select_gas_particles(ht, bhmass, rmax):
	print(rmax, " rmax!")
	cen_gas = ht.g[pynbody.filt.LowPass('r',rmax)]
	tot_mass = cen_gas['mass'].in_units('Msol').sum()
	if tot_mass < bhmass:
		raise RuntimeError('BH mass exceeds gas mass nearby!')
	osort = np.argsort(cen_gas['rho']) #sort by density
	osort = osort[::-1] #sort in descending order


	mgas_cumsum = np.cumsum(cen_gas['mass'][osort].in_units('Msol'))

	idelete = np.where(mgas_cumsum<=bhmass)[0]
	delete_iords = cen_gas['iord'][osort[idelete]]
	new_mass = -1
	new_mass_iord = -1

	still_needed = bhmass - mgas_cumsum[idelete[-1]]
	i = idelete[-1]+1
	if still_needed / cen_gas[osort[i]]['mass'].in_units('Msol') > 1:
		raise RuntimeError("Error! Gas should have already been deleted!")
	if still_needed / cen_gas[osort[i]]['mass'].in_units('Msol') > 0.8:
		delete_iords.append(cen_gas[osort[i]]['iord'])
	else:
		new_mass = pynbody.array.SimArray(cen_gas[osort[i]]['mass'].in_units('Msol') - still_needed, 'Msol')
		new_mass_iord = cen_gas[osort[i]]['iord']

	print(delete_iords, " gas particles have been selected for deletion with max distance ",
	      cen_gas['r'][osort[idelete]].in_units('kpc').max(), "kpc and minimum density ",
	      cen_gas['rho'][osort[idelete]].in_units('m_p cm**-3').min(), " mp/cm^3")
	print("gas particle iord", new_mass_iord, "selected for new mass ", new_mass, "from ",
	      cen_gas[osort[i]]['mass'].in_units('Msol'))

	return delete_iords, new_mass.in_units(ht.infer_original_units('Msol')), new_mass_iord





def create_bh_starlog(snap, sl, bhdata, filename):
	f = pynbody.util.open_(filename, 'wb')
	file_structure = get_starlog_meta(sl)
	if sl._byteswap:
		f.write(struct.pack(">i", file_structure.itemsize))
	else:
		f.write(struct.pack("i", file_structure.itemsize))

	sluse = np.where(np.in1d(sl['iord'], snap.s['iord']))[0]

	nstar = len(sluse)
	print("creating a starlog file for ", nstar, "stars plus ", len(bhdata['iord']), "BHs!")
	if nstar != len(snap.s):
		print("WARNING! STAR PARTICLE NUMBERS DON'T MATCH UP!")

	sldata = np.zeros(len(sluse)+len(bhdata['iord']), dtype=file_structure)
	for key in file_structure.names:
		sldata[key][:nstar] = sl[key][sluse]

	#add in bh data that's meaningful
	sldata['x'][nstar:] = bhdata['pos'].in_units(snap.infer_original_units(bhdata['pos'].units),
	                                             a = snap.properties['a'])[:,0].astype(file_structure['x'])
	sldata['y'][nstar:] = bhdata['pos'].in_units(snap.infer_original_units(bhdata['pos'].units),
	                                             a = snap.properties['a'])[:,1].astype(file_structure['y'])
	sldata['z'][nstar:] = bhdata['pos'].in_units(snap.infer_original_units(bhdata['pos'].units),
	                                             a = snap.properties['a'])[:,2].astype(file_structure['z'])
	sldata['vx'][nstar:] = bhdata['vel'].in_units(snap.infer_original_units(bhdata['vel'].units),
	                                             a = snap.properties['a'])[:, 0].astype(file_structure['vx'])
	sldata['vy'][nstar:] = bhdata['vel'].in_units(snap.infer_original_units(bhdata['vel'].units),
	                                             a = snap.properties['a'])[:, 1].astype(file_structure['vy'])
	sldata['vz'][nstar:] = bhdata['vel'].in_units(snap.infer_original_units(bhdata['vel'].units),
	                                             a = snap.properties['a'])[:, 2].astype(file_structure['vz'])
	sldata['massform'][nstar:] = bhdata['mass'].in_units(snap.infer_original_units(bhdata['mass'].units),
	                                             a = snap.properties['a']).astype(file_structure['massform'])
	sldata['iord'][nstar:] = bhdata['iord'].astype(file_structure['iord'])
	sldata['tform'][nstar:] = bhdata['tform'].in_units(snap.infer_original_units(bhdata['tform'].units),
	                                             a = snap.properties['a']).astype(file_structure['tform'])

	# add in meaningless BH data for filler space
	for key in file_structure.names:
		if key not in ['x','y','z','vx','vy','vz', 'massform','tform','iord']:
			sldata[key][nstar:] = np.zeros(len(bhdata['iord'])).astype(file_structure[key])

	print("writing starlog data...")
	if sl._byteswap:
		sldata.byteswap().tofile(f)
	else:
		sldata.tofile(f)
	f.close()
	return








		
		



