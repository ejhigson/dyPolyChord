#!/usr/bin/env python
"""
Prior functions.
"""
import PyPolyChord.priors


def uniform(hypercube, prior_scale=5):
    """Uniform prior."""
    ndims = len(hypercube)
    theta = [0.0] * ndims
    func = PyPolyChord.priors.UniformPrior(-prior_scale, prior_scale)
    for i, x in enumerate(hypercube):
        theta[i] = func(x)
    return theta


def gaussian(hypercube, prior_scale=5):
    """Spherically symmetric Gaussian prior centred on the origin. """
    ndims = len(hypercube)
    theta = [0.0] * ndims
    func = PyPolyChord.priors.GaussianPrior(0., prior_scale)
    for i, x in enumerate(hypercube):
        theta[i] = func(x)
    return theta
