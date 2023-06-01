import numpy as np

_default_dt = 0.02

def init_mdot(tend, dt):
	"""initialize a time array and an mdot array given a final time tend and time resolution dt"""
	tarray = np.arange(0,tend+dt, dt)
	return tarray, np.zeros(len(tarray))

def combine_mdot(raw_mdot_list, time, dt):
	"""combine a list of raw mdot_histogram outputs from tangos into a single array,
	working backwards in time such that overlapping timesteps are overwritten by
	earlier branches of the merger tree"""
	tarray, mdot_array = init_mdot(time.max(), dt)
	if len(raw_mdot_list) != len(time):
		raise RuntimeError("Time and Mdot data not the same length!")
	for i in range(len(raw_mdot_list)):
		tmin = time[i] - len(raw_mdot_list[i])*dt
		tmax = time[i]
		ind = np.where((tarray>=tmin) & (tarray <=tmax))[0]
		mdot_array[ind] = raw_mdot_list[i]

	return tarray, mdot_array

def trace_bh_galaxy(halo, bh_link_string):
	"""collect and combine raw BH histograms for selected BHs along merger tree"""
	mdot_hist_list, time_list = halo.calculate_for_progenitors(bh_link_string + '.raw(BH_mdot_histogram)', 't()')
	dt = halo.timestep.simulation.get('histogram_delta_t_Gyr', _default_dt)
	tarray, mdot_array = combine_mdot(mdot_hist_list, time_list, dt)
	return mdot_array, tarray

def gal_bh_acc_hist(halo, *constraints, selection='BH_mass', minmax='max'):
	"""Traces along a galaxy's main progenitor branch and connects indivitual BH accretion rate
	histograms along that tree, selected by BH property constraints at each timestep.

	For example, one might want to exctract the accretion rate for the most massive central (D<1 kpc)
	BH within a galaxy. If that BH went through a merger event, its individual accretion history might not
	be representative of the accretion history of the BH within the target galaxy because iords are
	arbitrarily kept/removed following BH merger events."""
	link_string = 'link(BH_central,'
	link_string += selection +','+'"'+minmax+'"'
	for i in range(len(constraints)):
		link_string += ', ' + constraints[i]
	link_string += ')'
	print("creating an accretion history for "+link_string+"in halo "+
	      str(halo.halo_number)+"at step "+halo.timestep.extension)

	mdot_array, tarray = trace_bh_galaxy(halo, link_string)
	return mdot_array, tarray

def most_massive_bh_acc_hist(halo):
	return gal_bh_acc_hist(halo, selection='BH_mass', minmax='max')

def central_bh_acc_hist(halo, max_dist=1):
	return gal_bh_acc_hist(halo, 'bh_central_distance<'+str(max_dist), selection='BH_mass', minmax='max')

def brightest_bh_acc_hist(halo):
	return gal_bh_acc_hist(halo, selection='BH_mdot_ave', minmax='max')

def brightest_central_bh_acc_hist(halo, max_dist=1):
	return gal_bh_acc_hist(halo, 'bh_central_distance<'+str(max_dist), selection='BH_mdot_ave', minmax='max')
