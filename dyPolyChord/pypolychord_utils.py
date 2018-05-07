#!/usr/bin/env python
"""
Functions for running dyPolyChord using PyPolyChord (PolyChord's built-in
python wrapper) using with python likelihoods and priors.
"""
# Exception handling needed to allow readthedocs to work without installing
# PyPolyChord.
try:
    import PyPolyChord
    import PyPolyChord.settings
except ImportError:
    pass


class RunPyPolyChord(object):

    """Callable class for running PolyChord in Python with specified
    the settings."""

    def __init__(self, likelihood, prior, ndim, nderived=0):
        """
        Specify likelihood, prior and number of dimensions in calculation.

        Parameters
        ----------
        likelihood: func
        prior: func
        ndim: int
        nderived: int, optional
        """
        self.likelihood = likelihood
        self.prior = prior
        self.ndim = ndim
        self.nderived = nderived

    def __call__(self, settings_dict, comm=None):
        """
        Runs PyPolyChord with specified inputs and writes output files. See the
        PyPolyChord documentation for more details.

        Parameters
        ----------
        settings_dict: dict
            Input PolyChord settings.
        comm: None or mpi4py MPI.COMM object, optional
            For MPI parallelisation.
        """
        if comm is None:
            settings = PyPolyChord.settings.PolyChordSettings(
                self.ndim, self.nderived, **settings_dict)
        else:
            rank = comm.Get_rank()
            if rank == 0:
                settings = PyPolyChord.settings.PolyChordSettings(
                    self.ndim, self.nderived, **settings_dict)
            else:
                settings = None
            settings = comm.bcast(settings, root=0)
        PyPolyChord.run_polychord(self.likelihood, self.ndim, self.nderived,
                                  settings, prior=self.prior)
