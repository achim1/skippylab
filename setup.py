from setuptools import setup

import os.path, re, sys


def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    with open(os.path.join(package, '__init__.py'), 'rb') as init_py:
        src = init_py.read().decode('utf-8')
        return re.search("__version__ = ['\"]([^'\"]+)['\"]", src).group(1)

version = get_version('skippylab')

tests_require = [
    'pytest>=3.0.5',
    'pytest-cov',
    'pytest-runner',
]

needs_pytest = set(('pytest', 'test', 'ptr')).intersection(sys.argv)
setup_requires = ['pytest-runner'] if needs_pytest else []

setup(name='skippylab',
      version=version,
      description='Readout instuments communication via vxi11 and SCPI like the TektronixDPO4104B oscilloscope',
      long_description='Use the oscilloscope for readout of waveforms bascially as a daq. Provides an easy to extend API to inlcude more functionality',
      author='Achim Stoessl',
      author_email="achim.stoessl@gmail.com",
      url='https://github.com/achim1/skippylab',
      #download_url="pip install skippylab",
      install_requires=['numpy>=1.11.0',
                        'matplotlib>=1.5.0',
                        'appdirs>=1.4.0',
                        'pyprind>=2.9.6',
                        'pyserial>=3.4.0,
                        'six>=1.1.0',
                        "python-vxi11>=0.9.0"],
      setup_requires=setup_requires,
      tests_require=tests_require,
      license="GPL",
      platforms=["Ubuntu 14.04","Ubuntu 16.04"],
      classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Physics"
              ],
      keywords=["oscilloscope", "daq",\
                "TektronixDPO4104B", "Tektronix",\
                "readout", "physics", "engineering", "SCPI", "VISA", "vxi11"],
      packages=['skippylab', 'skippylab.instruments', 'skippylab.scpi'],
      #scripts=[],
      #package_data={'skippylab': ['pyoscidefault.mplstyle','pyoscipresent.mplstyle']}
      )
