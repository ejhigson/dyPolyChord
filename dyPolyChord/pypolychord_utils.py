#!/usr/bin/env python
"""
Run dyPolyChord using PolyChord's python wrapper with python likelihoods and
priors.
"""
import functools
import PyPolyChord
import PyPolyChord.settings


def python_run_func(settings_dict, likelihood=None, prior=None, ndims=None,
                    nderived=0):
    """python_run_func."""
    settings = PyPolyChord.settings.PolyChordSettings(
        ndims, nderived, **settings_dict)
    return PyPolyChord.run_polychord(likelihood, ndims, nderived, settings,
                                     prior)


def get_python_run_func(likelihood, prior, ndims, nderived=0):
    """Utility function for freezing python run func args."""
    return functools.partial(python_run_func, ndims=ndims,
                             likelihood=likelihood, prior=prior,
                             nderived=nderived)
