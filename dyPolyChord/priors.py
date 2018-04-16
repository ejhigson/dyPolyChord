#!/usr/bin/env python
"""
Prior functions.

Input hypercube values are mapped to physical space using the inverse CDF
(cumulative distribution function) of each parameter.
See the PolyChord papers for more details.
"""
import PyPolyChord.priors


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
    ndims = len(hypercube)
    theta = [0.0] * ndims
    func = PyPolyChord.priors.UniformPrior(-prior_scale, prior_scale)
    for i, x in enumerate(hypercube):
        theta[i] = func(x)
    return theta


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
    ndims = len(hypercube)
    theta = [0.0] * ndims
    func = PyPolyChord.priors.GaussianPrior(0., prior_scale)
    for i, x in enumerate(hypercube):
        theta[i] = func(x)
    return theta
