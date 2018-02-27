#!/usr/bin/env python
"""dypypolychord setup."""
import os
import setuptools


def read_file(fname):
    """
    For using the README file as the long description.
    From https://pythonhosted.org/an_example_pypi_project/setuptools.html
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(name='dypypolychord',
                 version='0.0.0',
                 author='Edward Higson',
                 author_email='ejhigson@gmail.com',
                 description=('Dynamic nested sampling with PolyChord using '
                              'python wrappers.'),
                 url='https://github.com/ejhigson/nestcheck',
                 # long_description=read_file('README.md'),
                 # install_requires=['numpy>=1.13',
                 #                   'scipy>=0.18.1',
                 #                   'nestcheck',
                 #                   'PyPolyChord'],
                 # Add minimum PyPolyChord version to install_requires when
                 # it is publically released
                 # test_suite='nose.collector',
                 # tests_require=['nose'],
                 packages=['dypypolychord'])
