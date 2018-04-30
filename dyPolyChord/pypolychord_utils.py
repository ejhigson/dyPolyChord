#!/usr/bin/env python
"""
Functions for running dyPolyChord using PyPolyChord (PolyChord's built-in
python wrapper) using with python likelihoods and priors.
"""
import functools
import PyPolyChord
import PyPolyChord.settings


def python_run_func(settings_dict, **kwargs):
    """
    Runs PyPolyChord with specified inputs and writes output files. See the
    PyPolyChord documentation for more details.

    Parameters
    ----------
    settings_dict: dict
        Input PolyChord settings.
    likelihood: func
        Loglikelihood function (see the PolyChord documentation for more
        details).
    prior: func
        Prior function mapping from hypercube to physical space (see the
        PolyChord documentation for more details).
    ndim: int
        Number of parameters.
    nderived: int, optional
        Number of derived parameters
    comm: None or mpi4py MPI.COMM object, optional
        For MPI parallelisation.
    """
    likelihood = kwargs.pop('likelihood')
    prior = kwargs.pop('prior')
    ndim = kwargs.pop('ndim')
    nderived = kwargs.pop('nderived', 0)
    comm = kwargs.pop('comm', None)
    if comm is None:
        settings = PyPolyChord.settings.PolyChordSettings(
            ndim, nderived, **settings_dict)
    else:
        rank = comm.Get_rank()
        if rank == 0:
            settings = PyPolyChord.settings.PolyChordSettings(
                ndim, nderived, **settings_dict)
        else:
            settings = None
        settings = comm.bcast(settings, root=0)
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
