#!/usr/bin/env python
"""
Prior functions.
"""
import PyPolyChord.priors


def uniform_prior(hypercube, ndims=2, prior_scale=5):
    """ Uniform prior. """

    theta = [0.0] * ndims
    uniform = PyPolyChord.priors.UniformPrior(-prior_scale, prior_scale)
    for i, x in enumerate(hypercube):
        theta[i] = uniform(x)

    return theta


def gaussian_prior(hypercube, ndims=2, prior_scale=5):
    """Gaussian prior centred on the origin. """

    theta = [0.0] * ndims
    uniform = PyPolyChord.priors.GaussianPrior(0., prior_scale)
    for i, x in enumerate(hypercube):
        theta[i] = uniform(x)

    return theta
