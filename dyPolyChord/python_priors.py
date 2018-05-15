#!/usr/bin/env python
"""
Python priors for use with PolyChord.

PolyChord v1.14 requires priors to be callables with parameter and return
types:

Parameters
----------
hypercube: float or 1d numpy array
    Parameter positions in the prior hypercube.

Returns
-------
theta: float or 1d numpy array
    Corresponding physical parameter coordinates.

Input hypercube values numpy array is mapped to physical space using the
inverse CDF (cumulative distribution function) of each parameter.
See the PolyChord papers for more details.

We use classes with the prior defined in the __call__ property, as
this provides convenient way of storing other information such as
hyperparameter values. The objects be used in the same way as functions
due to python's "duck typing" (or alternatively you can just define prior
functions).
"""
import numpy as np
import scipy


class Uniform(object):

    """Uniform prior."""

    def __init__(self, minimum=0.0, maximum=1.0):
        """
        Set up prior object's hyperparameter values.

        Prior is uniform in [minimum, maximum] in each parameter.

        Parameters
        ----------
        minimum: float
        maximum: float
        """
        assert maximum > minimum
        self.maximum = maximum
        self.minimum = minimum

    def __call__(self, hypercube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: list of floats
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: list of floats
            Physical parameter values corresponding to hypercube.
        """
        return self.minimum + (self.maximum - self.minimum) * hypercube


class Gaussian(object):

    """Symmetric Gaussian prior centred on the origin."""

    def __init__(self, sigma=10.0):
        """
        Set up prior object's hyperparameter values.

        Parameters
        ----------
        sigma: float
            Standard deviation of Gaussian prior in each parameter.
        """
        self.sigma = sigma

    def __call__(self, hypercube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: list of floats
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: list of floats
            Physical parameter values corresponding to hypercube.
        """
        theta = scipy.special.erfinv(2 * hypercube - 1)
        theta *= self.sigma * np.sqrt(2)
        return theta
