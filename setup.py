#!/usr/bin/env python
"""
dyPolyChord setup module.

Based on https://github.com/pypa/sampleproject/blob/master/setup.py.
"""
import os
import setuptools


def get_package_dir():
    """Get the file path for dyPolyChord's directory."""
    return os.path.abspath(os.path.dirname(__file__))


def get_long_description():
    """Get PyPI long description from README.rst."""
    pkg_dir = get_package_dir()
    with open(os.path.join(pkg_dir, 'README.rst')) as readme_file:
        long_description = readme_file.read()
    return long_description


def get_version():
    """Get single-source __version__."""
    pkg_dir = get_package_dir()
    with open(os.path.join(pkg_dir, 'dyPolyChord/_version.py')) as ver_file:
        string = ver_file.read()
    return string.strip().replace('__version__ = ', '').replace('\'', '')


setuptools.setup(name='dyPolyChord',
                 version=get_version(),
                 description=(
                     'Super fast dynamic nested sampling with '
                     'PolyChord (python, C++ and Fortran likelihoods).'),
                 long_description=get_long_description(),
                 long_description_content_type='text/x-rst',
                 url='https://github.com/ejhigson/dyPolyChord',
                 author='Edward Higson',
                 author_email='e.higson@mrao.cam.ac.uk',
                 license='MIT',
                 keywords='nested-sampling dynamic-nested-sampling',
                 classifiers=[  # Optional
                     'Development Status :: 5 - Production/Stable',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 2',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Topic :: Scientific/Engineering :: Astronomy',
                     'Topic :: Scientific/Engineering :: Physics',
                     'Topic :: Scientific/Engineering :: Information Analysis',
                 ],
                 packages=['dyPolyChord'],
                 # Note that PolyChord is also required to do nested sampling
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'nestcheck>=0.1.8'],
                 test_suite='nose.collector',
                 tests_require=['nose', 'coverage'],
                 extras_require={'docs': ['sphinx',
                                          'numpydoc',
                                          'sphinx-rtd-theme',
                                          'nbsphinx>=0.3.3'],
                                 'MPI': ['mpi4py']},
                 project_urls={  # Optional
                     'Docs': 'http://dyPolyChord.readthedocs.io/en/latest/'})
