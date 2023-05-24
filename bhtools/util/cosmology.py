import numpy as np
import pynbody
import scipy
import scipy.integrate
import scipy.optimize
from scipy.interpolate import interp1d

_interp_points = 1000

def getTime(z, h0, omegaM0, omegaL0, unit='Gyr'):
	def _a_dot(a, h0, om_m, om_l):
		om_k = 1.0 - om_m - om_l
		return h0 * a * np.sqrt(om_m * (a ** -3) + om_k * (a ** -2) + om_l)

	def _a_dot_recip(*args):
		return 1. / _a_dot(*args)

	conv = pynbody.units.Unit("0.01 s Mpc km^-1").ratio(unit)

	def get_age(x):
		x = 1.0 / (1.0 + x)
		return scipy.integrate.quad(_a_dot_recip, 0, x, (h0, omegaM0, omegaL0))[0] * conv

	if isinstance(z, np.ndarray) or isinstance(z, list):
		if len(z) > _interp_points:
			a_vals = np.logspace(-3, 0, _interp_points)
			z_vals = 1. / a_vals - 1.
			log_age_vals = np.log(getTime(z_vals, h0, omegaM0, omegaL0, unit=unit))
			interp = interp1d(np.log(a_vals), log_age_vals, bounds_error=False)
			log_a_input = np.log(1. / (1. + z))
			results = np.exp(interp(log_a_input))
		else:
			results = np.array(list(map(get_age, z)))
		results = results.view(pynbody.array.SimArray)
		results.units = unit
	else:
		results = get_age(z)
	return results

def getRedshift(times, h0, omegaM0, omegaL0, verbose=False):
		redshift = np.zeros(np.size(times))
		ntimes = np.size(times)
		if ntimes == 1 and hasattr(times,'__getitem__')==False:
			print('here')
			times = np.array([times])
		if hasattr(times,'units')==False:
			print("WARNING assuming provided time is in units of Gyr")
			times = pynbody.array.SimArray(times,'Gyr')
		for tt in range(np.size(times)):
				def func(z):
						return getTime(z, h0, omegaM0, omegaL0, unit='Gyr') - times.in_units('Gyr')[tt]

				try: redshift[tt] = scipy.optimize.newton(func, 0)
				except:
					try:
						print("trying again")
						redshift[tt] = scipy.optimize.newton(func, 20)
					except:
						print(("ERROR did not converge", times[tt], tt))
						redshift[tt] = -1
		return redshift
