from distutils.core import setup


DESCRIPTION = 'Tools to help analyze black holes in large nbody simulations (in conjunction with pynbody)'
LONG_DESCRIPTION = open('README.md').read()
NAME = 'bhtools'
VERSION = '1.0'
AUTHOR = 'Michael Tremmel'
AUTHOR_EMAIL = 'm.tremmel6@gmail.com'
MAINTAINER = 'Michael Tremmel'
MAINTAINER_EMAIL = 'm.tremmel6@gmail.com'
URL = ''
DOWNLOAD_URL = ''
LICENSE = 'BSD'



setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email= AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      package_dir={'/':''},
      packages=['bhtools',
                'bhtools/util',
                'bhtools/output_reader',],
      package_data={},
	  classifiers=["Development Status :: 3 - Alpha",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                     "Programming Language :: Python :: 3",
                     "Topic :: Scientific/Engineering :: Astronomy"])