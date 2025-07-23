import pynbody
import math

class ParamFile(object):
	"""
	Object that parses the param file and determins base units for a simulation
	Much of this is based of off pynbody's code
	"""
	def __init__(self, filename):
		try:
			f = open(filename)
		except IOError:
			raise IOError("The parameter filename you supplied is invalid")

		if f is None:
			return

		self.params = {}

		for line in f:
			if line[0] != "#" and len(line)>0:
				s = line.split("#")[0].split()
				self.params[s[0]] = " ".join(s[2:])
		f.close()
		self._get_basic_units()
		self._get_cosmology()
		if 'dInitBHMass' in self.params.keys():
			self.init_mbh = float(self.params['dInitBHMass'])*self.munit
		else:
			self.init_mbh = -1

	def __getitem__(self, item):
		return self.params[item]

	def _get_cosmology(self):
		self.omegaM = float(self.params['dOmega0'])
		self.omegaL = float(self.params['dLambda'])
		self.h = float(self.params['dHubble0'])
		hubunit = 10. * self.velunit / self.dunit
		self.h *= hubunit

	def _get_basic_units(self):
		self.munit_st = pynbody.units.Unit(self.params["dMsolUnit"] + " Msol")
		self.munit = float(self.params["dMsolUnit"])
		self.dunit_st = pynbody.units.Unit(self.params["dKpcUnit"] + " kpc")
		self.dunit = float(self.params["dKpcUnit"])

		self.denunit = self.munit / self.dunit ** 3
		self.denunit_st = pynbody.units.Unit(str(self.denunit) + " Msol kpc^-3")

		# denunit_cgs = denunit * 6.7696e-32
		# kpc_in_km = 3.0857e16
		# secunit = 1./math.sqrt(denunit_cgs*6.67e-8)
		# velunit = dunit*kpc_in_km/secunit
		# The following (from pynbody) avoids any numerical innacuracies

		self.velunit = 8.0285 * math.sqrt(6.6743e-8 * self.denunit) * self.dunit
		self.velunit_st = pynbody.units.Unit(("%.5g" % self.velunit) + " km s^-1")

		self.timeunit = self.dunit / self.velunit * 0.97781311
		self.timeunit_st = pynbody.units.Unit(("%.5g" % self.timeunit) + " Gyr")

		self.potunit_st = pynbody.units.Unit("%.5g km^2 s^-2" % (self.velunit ** 2))

		#if the simulation is cosmological and co-moving, put in the appropriate a's
		if 'bComove' in self.params and int(self.params['bComove']) != 0:
			self.dunit_st *= pynbody.units.a
			self.denunit *= pynbody.units.a**-3
			self.velunit_st *= pynbody.units.a
			self.potunit_st *= pynbody.units.a**-1

	def get_boxsize(self):
		'''
		:return: a unit object based on the kpc units of the simulation in the param file
		returned in units of co-moving Mpc
		'''
		return pynbody.units.Unit(self.params['dKpcUnit']+' a kpc')
