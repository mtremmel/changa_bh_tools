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

#useful functions below showing common examples
def most_massive_bh_acc_hist(halo):
    return gal_bh_acc_hist(halo, selection='BH_mass', minmax='max')

def central_bh_acc_hist(halo, max_dist=1):
    return gal_bh_acc_hist(halo, 'BH_central_distance<'+str(max_dist), selection='BH_mass', minmax='max')

def brightest_bh_acc_hist(halo):
    return gal_bh_acc_hist(halo, selection='BH_mdot_ave', minmax='max')

def brightest_central_bh_acc_hist(halo, max_dist=1):
    return gal_bh_acc_hist(halo, 'BH_central_distance<'+str(max_dist), selection='BH_mdot_ave', minmax='max')

def get_macc(bhdb, tform1, tform2):
    """
    Get the accreted mass for a specific BH, excluding all formation accretion events
    :param bhdb: simulation object for black hole
    :param tform1/tform2: together, all of the formation times
    for all progenitor BHs along the target BH's merger tree
    :param sim: name of simulatino in database
    :return: total accreted mass along complete merger tree for each BH
    """
    macc = -1
    mdot_all = bhdb.calculate('reassemble(BH_mdot_histogram,"sum")')
    tbad = np.concatenate([tform1, tform2])
    tbad = np.unique(tbad)
    obad = []
    use = np.ones(len(mdot_all))
    tmdot = np.arange(len(mdot_all)) * 0.01
    for j in range(len(tbad)):
        badstep = np.where((np.abs(tmdot - tbad[j]) < 0.005)&(mdot_all>=2e5/(0.01*1e9)))[0]
        if len(badstep) > 0: obad.extend(badstep)
    non_zero = np.where(mdot_all > 0)[0]
    if len(non_zero)>0:
        iform = np.where(mdot_all > 0)[0][0]  # take away first nonzero accretion output (affected by current BH formation)
        if iform not in obad:
            obad.append(iform)
        use[obad] = 0
        use_mdot = np.where(use == 1)[0]
        if len(use_mdot)>0:
            macc = np.nansum(mdot_all[use_mdot]) * 0.01 * 1e9
    else:
        macc = 0
    if len(np.where(mdot_all<0)[0])>0:
        raise RuntimeError("Negative accretion!")
    return macc

def get_macc_all(step, link_string, merger_data):
    '''
    Get accreted mass for all BHs in a given step based on some selection criteria
    :param step: simulation step (tangos timestep)
    :param link_string: string to select bhs from galaxies e.g. link(BH_central, BH_mass, "max")
    :param merger_data: data on BH mergers
    :return: bhids, array of all BH accreted masses from each galay given link string
    '''
    target_bhs, = step.calculate_all(link_string)
    bh_iord_list = []
    for i in range(len(target_bhs)):
        bh_iord_list.append(target_bhs[i].halo_number)
    from ..output_reader import mergers
    nmerge, tmerge, iord_merge, iord_merge_all, tlast, hid_bh, tform1, tform2, nmerge_all = \
        mergers.collect_all_bh_mergers(bh_iord_list, step.time_gyr,merger_data, 0.0,0.0)
    if len(np.where(bh_iord_list!=hid_bh)[0])>0:
        raise RuntimeError("Black Hole IDs not matching up!")
    macc_all = np.ones(len(target_bhs))*-1 #initialize to -1
    for i in range(len(target_bhs)):
        if i%100==0:
            print(i/len(target_bhs))
        macc_all[i] = get_macc(target_bhs[i],tform1[i], tform2[i])
    return np.array(bh_iord_list), macc_all
