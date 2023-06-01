import numpy as np
import matplotlib.pyplot as plt

class MbhMstarRelation(object):
    #parameters are a, b, c, and intrinsic scatter
    #general relation is logMbh = c+ a*lobMstar - b
    def __init__(self):
        self._params = {'SS2013':(1.12, 11, 8.31, 0.3),
                    'RV15AGN': (1.05, 11, 7.75, 0.24)}

    def _calc_relation(self, logMstar, citation):
        p = self._params[citation]
        return p[2]+ p[0]*(logMstar - p[1])

    def intrinsic_scatter(self, citation):
        return self._params[citation][4]

    def plot(self, citation, logmstar_min, logmstar_max, log=False, color='k', show_scatter=True, scatter_alpha=0.5, **kwargs):
        xarray  = np.arange(logmstar_min,logmstar_max, 0.01)
        yarray = self._calc_relation(xarray, citation)
        if not log:
            xarray = 10**xarray
            yarray = 10**yarray

        plt.plot(xarray, yarray, color=color, **kwargs)
        if show_scatter:
            if not log:
                plt.fill_between(xarray, 10**(np.log10(yarray)+self._params[citation][3]),
                             10**(np.log10(yarray)-self._params[citation][3]), alpha=scatter_alpha, color=color)
            else:
                plt.fill_between(xarray, yarray+self._params[citation][3], yarray-self._params[citation][3],
                                 alpha=scatter_alpha, color=color)