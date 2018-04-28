#!/usr/bin/env python
"""
Functions for running dyPolyChord using PyPolyChord (PolyChord's built-in
python wrapper) using with python likelihoods and priors.
"""
import functools
import PyPolyChord
import PyPolyChord.settings


def python_run_func(settings_dict, likelihood=None, prior=None, ndim=None,
                    nderived=0):
    """
    Runs PyPolyChord with specified inputs and writes output files. See the
    PyPolyChord documentation for more details.

    Parameters
    ----------
    settings_dict: dict
        Input PolyChord settings.
    likelihood: func
    prior: func
    ndim: int
    nderived: int
    """
    settings = PyPolyChord.settings.PolyChordSettings(
        ndim, nderived, **settings_dict)
    PyPolyChord.run_polychord(likelihood, ndim, nderived, settings, prior)


def get_python_run_func(likelihood, prior, ndim, nderived=0):
    """
    Helper function for freezing python_run_func args.

    Parameters
    ----------
    likelihood: func
    prior: func
    ndim: int
    nderived: int, optional

    Returns
    -------
    functools partial object
        python_run_func with input parameters frozen.
    """
    return functools.partial(python_run_func, ndim=ndim,
                             likelihood=likelihood, prior=prior,
                             nderived=nderived)
