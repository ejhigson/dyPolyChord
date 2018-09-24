#!/usr/bin/env python
"""Python priors for use with PolyChord.

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

This module use classes with the prior defined in the cube_to_physical
function and called with __call__ property. This provides convenient way of
storing other information such as hyperparameter values. The objects be
used in the same way as functions due to python's "duck typing" (or
alternatively you can just define prior functions).

The BlockPrior class allows convenient use of different priors on different
parameters.

Inheritance of the BasePrior class allows priors to:

   1. have parameters' values sorted to give an enforced order. Useful when the
   parameter space is symmetric under exchange of variables as this allows the
   space to be explored to be contracted by a factor of N! (where N is the
   number of such parameters);

   2. adaptively select the number of parameters to use.

You can ignore these if you don't need them.
"""
import copy
import numpy as np
import scipy


class BasePrior(object):

    """Base class for Priors."""

    def __init__(self, adaptive=False, sort=False, nfunc_min=1):
        """
        Set up prior object's hyperparameter values.

        Parameters
        ----------
        adaptive: bool, optional
        sort: bool, optional
        nfunc_min: int, optional
        """
        self.adaptive = adaptive
        self.sort = sort
        self.nfunc_min = nfunc_min

    def cube_to_physical(self, cube):  # pylint: disable=no-self-use
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        cube: 1d numpy array
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: 1d numpy array
            Physical parameter values corresponding to hypercube.
        """
        return cube

    def __call__(self, cube):
        """
        Evaluate prior on hypercube coordinates.

        Parameters
        ----------
        cube: 1d numpy array
            Point coordinate on unit hypercube (in probabily space).
            Note this variable cannot be edited else PolyChord throws an error.

        Returns
        -------
        theta: 1d numpy array
            Physical parameter values for prior.
        """
        if self.adaptive:
            try:
                theta = adaptive_transform(
                    cube, sort=self.sort, nfunc_min=self.nfunc_min)
            except ValueError:
                if np.isnan(cube[0]):
                    return np.full(cube.shape, np.nan)
                else:
                    raise
            theta[1:] = self.cube_to_physical(theta[1:])
            return theta
        else:
            if self.sort:
                return self.cube_to_physical(forced_identifiability(cube))
            else:
                return self.cube_to_physical(cube)


class Gaussian(BasePrior):

    """Symmetric Gaussian prior centred on the origin."""

    def __init__(self, sigma=10.0, half=False, mu=0.0, **kwargs):
        """
        Set up prior object's hyperparameter values.

        Parameters
        ----------
        sigma: float
            Standard deviation of Gaussian prior in each parameter.
        half: bool, optional
            Half-Gaussian prior - nonzero only in the region greater than mu.
            Note that in this case mu is no longer the mean and sigma is no
            longer the standard deviation of the prior.
        mu: float, optional
            Mean of Gaussian prior.
        kwargs: dict, optional
            See BasePrior.__init__ for more infomation.
        """
        BasePrior.__init__(self, **kwargs)
        self.sigma = sigma
        self.mu = mu
        self.half = half

    def cube_to_physical(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        cube: 1d numpy array
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: 1d numpy array
            Physical parameter values corresponding to hypercube.
        """
        if self.half:
            theta = scipy.special.erfinv(cube)
        else:
            theta = scipy.special.erfinv(2 * cube - 1)
        theta *= self.sigma * np.sqrt(2)
        return theta + self.mu


class Uniform(BasePrior):

    """Uniform prior."""

    def __init__(self, minimum=0.0, maximum=1.0, **kwargs):
        """
        Set up prior object's hyperparameter values.

        Prior is uniform in [minimum, maximum] in each parameter.

        Parameters
        ----------
        minimum: float
        maximum: float
        kwargs: dict, optional
            See BasePrior.__init__ for more infomation.
        """
        BasePrior.__init__(self, **kwargs)
        assert maximum > minimum, (minimum, maximum)
        self.maximum = maximum
        self.minimum = minimum

    def cube_to_physical(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        cube: 1d numpy array

        Returns
        -------
        theta: 1d numpy array
        """
        return self.minimum + (self.maximum - self.minimum) * cube


class PowerUniform(BasePrior):

    """Uniform in theta ** power"""

    def __init__(self, minimum=0.1, maximum=2.0, power=-2, **kwargs):
        """
        Set up prior object's hyperparameter values.

        Prior is uniform in [minimum, maximum] in each parameter.

        Parameters
        ----------
        minimum: float
        maximum: float
        power: float or int
        kwargs: dict, optional
            See BasePrior.__init__ for more infomation.
        """
        BasePrior.__init__(self, **kwargs)
        assert maximum > minimum > 0, (minimum, maximum)
        assert power != 0, power
        self.maximum = maximum
        self.minimum = minimum
        self.power = power
        self.const = abs((minimum ** (1. / power)) - (maximum ** (1. / power)))
        self.const = 1 / self.const

    def cube_to_physical(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        cube: 1d numpy array

        Returns
        -------
        theta: 1d numpy array
        """
        if self.power > 0:
            theta = (self.minimum ** (1. / self.power)) + (cube / self.const)
        else:
            theta = (self.minimum ** (1. / self.power)) - (cube / self.const)
        return theta ** self.power


class Exponential(BasePrior):

    """Exponential prior."""

    def __init__(self, lambd=1.0, **kwargs):
        """
        Set up prior object's hyperparameter values.

        Parameters
        ----------
        lambd: float
        kwargs: dict, optional
            See BasePrior.__init__ for more infomation.
        """
        BasePrior.__init__(self, **kwargs)
        self.lambd = lambd

    def cube_to_physical(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        cube: 1d numpy array

        Returns
        -------
        theta: 1d numpy array
        """
        return - np.log(1 - cube) / self.lambd


class BlockPrior(object):

    """Prior object which applies a list of priors to different blocks within
    the parameters."""

    def __init__(self, prior_blocks, block_sizes):
        """Store prior and size of each block."""
        assert len(prior_blocks) == len(block_sizes), (
            'len(prior_blocks)={}, len(block_sizes)={}, block_sizes={}'
            .format(len(prior_blocks), len(block_sizes), block_sizes))
        self.prior_blocks = prior_blocks
        self.block_sizes = block_sizes

    def __call__(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: 1d numpy array
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: 1d numpy array
            Physical parameter values corresponding to hypercube.
        """
        theta = np.zeros(cube.shape)
        start = 0
        end = 0
        for i, prior in enumerate(self.prior_blocks):
            end += self.block_sizes[i]
            theta[start:end] = prior(cube[start:end])
            start += self.block_sizes[i]
        return theta


# Helper functions
# ----------------


def forced_identifiability(cube):
    """Transform hypercube coordinates to enforce identifiability.

    For more details see: "PolyChord: next-generation nested sampling"
    (Handley et al. 2015).

    Parameters
    ----------
    cube: 1d numpy array
        Point coordinate on unit hypercube (in probabily space).

    Returns
    -------
    ordered_cube: 1d numpy array
    """
    ordered_cube = np.zeros(cube.shape)
    ordered_cube[-1] = cube[-1] ** (1. / cube.shape[0])
    for n in range(cube.shape[0] - 2, -1, -1):
        ordered_cube[n] = cube[n] ** (1. / (n + 1)) * ordered_cube[n + 1]
    return ordered_cube


def adaptive_transform(cube, sort=True, nfunc_min=1):
    """Tranform first parameter (nfunc) to uniform in (nfunc_min, nfunc_max)
    and, if required, perform forced identifiability transform on the next
    nfunc parameters only.

    Parameters
    ----------
    cube: 1d numpy array
        Point coordinate on unit hypercube (in probabily space).

    Returns
    -------
    ad_cube: 1d numpy array
        First element is physical coordinate of nfunc parameter, other elements
        are cube coordinates with any forced identifiability transform already
        applied.
    """
    # First get integer number of funcs
    ad_cube = copy.deepcopy(cube)
    nfunc_max = cube.shape[0] - 1
    # first component is a number of funcs
    ad_cube[0] = ((nfunc_min - 0.5) + (1.0 + nfunc_max - nfunc_min) * cube[0])
    if sort:
        nfunc = int(np.round(ad_cube[0]))
        # Sort only parameters 1 to nfunc
        ad_cube[1:1 + nfunc] = forced_identifiability(cube[1:1 + nfunc])
    return ad_cube
