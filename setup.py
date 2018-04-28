#!/usr/bin/env python
"""
dyPolyChord setup module.

Based on https://github.com/pypa/sampleproject/blob/master/setup.py.
"""
import os
import setuptools


def get_long_description():
    """Get PyPI long description from README.rst."""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.rst')) as readme_file:
        long_description = readme_file.read()
    return long_description


setuptools.setup(name='dyPolyChord',
                 version='0.0.0',
                 description=(
                     'Blazingly fast dynamic nested sampling with '
                     'PolyChord (python, C++ and Fortran likelihoods).'),
                 long_description=get_long_description(),
                 long_description_content_type='text/x-rst',
                 url='https://github.com/ejhigson/dyPolyChord',
                 author='Edward Higson',
                 author_email='ejhigson@gmail.com',
                 license='MIT',
                 keywords='nested-sampling dynamic-nested-sampling',
                 classifiers=[  # Optional
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: MIT License',
                     # 'Programming Language :: Python :: 2',
                     # 'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'Programming Language :: Python :: 3.4',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Topic :: Scientific/Engineering :: Astronomy',
                     'Topic :: Scientific/Engineering :: Physics',
                     'Topic :: Scientific/Engineering :: Information Analysis',
                 ],
                 packages=['dyPolyChord'],
                 # futures is in standard library for python >= 3.2, but
                 # include it for compatibility with python 2.7.
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   # 'futures',
                                   'nestcheck'],
                 test_suite='nose.collector',
                 tests_require=['nose', 'coverage'],
                 extras_require={'docs': ['sphinx',
                                          'numpydoc',
                                          'sphinx-rtd-theme',
                                          'nbsphinx>=0.3.3']},
                 project_urls={  # Optional
                     'Docs': 'http://dyPolyChord.readthedocs.io/en/latest/'})
