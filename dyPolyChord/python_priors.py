#!/usr/bin/env python
"""
Prior functions.

Input hypercube values are mapped to physical space using the inverse CDF
(cumulative distribution function) of each parameter.
See the PolyChord papers for more details.
"""
import numpy as np
import scipy


def uniform(hypercube, prior_scale=5):
    """
    Uniform prior.

    Parameters
    ----------
    hypercube: list of floats
        Point coordinate on unit hypercube (in probabily space).
        See the PolyChord papers for more details.
    prior_scale: float
        prior is uniform [-prior_scale, +prior_scale] in each dimension.

    Returns
    -------
    theta: list of floats
        Physical parameter values corresponding to hypercube.
    """
    return [(-prior_scale + 2 * prior_scale * x) for x in hypercube]


def gaussian(hypercube, prior_scale=5):
    """
    Symmetric Gaussian prior centred on the origin.

    Parameters
    ----------
    hypercube: list of floats
        Point coordinate on unit hypercube (in probabily space).
        See the PolyChord papers for more details.
    prior_scale: float
        Standard deviation of Gaussian prior in each parameter..

    Returns
    -------
    theta: list of floats
        Physical parameter values corresponding to hypercube.
    """
    return [(prior_scale * np.sqrt(2) * scipy.special.erfinv(2 * x - 1))
            for x in hypercube]
