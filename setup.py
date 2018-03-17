#!/usr/bin/env python
"""dyPolyChord setup."""
import os
import setuptools


def read_file(fname):
    """
    For using the README file as the long description.
    From https://pythonhosted.org/an_example_pypi_project/setuptools.html
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setuptools.setup(name='dyPolyChord',
                 version='0.0.0',
                 author='Edward Higson',
                 author_email='ejhigson@gmail.com',
                 description=('Dynamic nested sampling with PolyChord.'),
                 url='https://github.com/ejhigson/dyPolyChord',
                 long_description=read_file('README.md'),
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'nestcheck',
                                   'PyPolyChord>=1.12'],
                 test_suite='nose.collector',
                 tests_require=['nose'],
                 packages=['dyPolyChord'])
