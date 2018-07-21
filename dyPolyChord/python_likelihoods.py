#!/usr/bin/env python
"""
Loglikelihood functions for use with PyPolyChord (PolyChord's python wrapper).

PolyChord v1.14 requires likelihoods to be callables with parameter and return
types:

Parameters
----------
theta: float or 1d numpy array

Returns
-------
logl: float
    Loglikelihood.
phi: list of length nderived
    Any derived parameters.

We use classes with the loglikelihood defined in the __call__ property, as
this provides convenient way of storing other information such as
hyperparameter values. These objects can be used in the same way as functions
due to python's "duck typing" (alternatively you can define likelihoods
using functions).
"""
import copy
import numpy as np
import scipy.special


class Gaussian(object):

    """Symmetric Gaussian loglikelihood centered on the origin."""

    def __init__(self, sigma=1.0, nderived=0):
        """
        Set up likelihood object's hyperparameter values.

        Parameters
        ----------
        sigma: float, optional
            Standard deviation of Gaussian (the same for each parameter).
        nderived: int, optional
            Number of derived parameters.
        """
        self.sigma = sigma
        self.nderived = nderived

    def __call__(self, theta):
        """
        Calculate loglikelihood(theta), as well as any derived parameters.

        Parameters
        ----------
        theta: float or 1d numpy array

        Returns
        -------
        logl: float
            Loglikelihood
        phi: list of length nderived
            Any derived parameters.
        """
        logl = log_gaussian_pdf(theta, sigma=self.sigma, mu=0)
        return logl, [0.0] * self.nderived


class GaussianShell(object):

    """Gaussian Shell loglikelihood centred on the origin."""

    def __init__(self, sigma=0.2, rshell=2, nderived=0):
        """
        Set up likelihood object's hyperparameter values.

        Parameters
        ----------
        sigma: float, optional
            Standard deviation of Gaussian (the same for each parameter).
        rshell: float, optional
            Distance of shell peak from origin.
        nderived: int, optional
            Number of derived parameters.
        """
        self.sigma = sigma
        self.rshell = rshell
        self.nderived = nderived

    def __call__(self, theta):
        """
        Calculate loglikelihood(theta), as well as any derived parameters.

        N.B. this loglikelihood is not normalised.

        Parameters
        ----------
        theta: float or 1d numpy array

        Returns
        -------
        logl: float
            Loglikelihood
        phi: list of length nderived
            Any derived parameters.
        """
        rad = np.sqrt(np.sum(theta ** 2))
        logl = - ((rad - self.rshell) ** 2) / (2 * self.sigma ** 2)
        return logl, [0.0] * self.nderived


class Rastrigin(object):

    """Rastrigin loglikelihood as described in the PolyChord paper."""

    def __init__(self, a=10, nderived=0):
        """
        Set up likelihood object's hyperparameter values.

        Parameters
        ----------
        a: float, optional
        nderived: int, optional
            Number of derived parameters.
        """
        self.a = a
        self.nderived = nderived

    def __call__(self, theta):
        """
        Calculate loglikelihood(theta), as well as any derived parameters.

        N.B. this loglikelihood is not normalised.

        Parameters
        ----------
        theta: float or 1d numpy array

        Returns
        -------
        logl: float
            Loglikelihood
        phi: list of length nderived
            Any derived parameters.
        """
        ftheta = self.a * len(theta)
        for th in theta:
            ftheta += (th ** 2) - self.a * np.cos(2 * np.pi * th)
        logl = -ftheta
        return logl, [0.0] * self.nderived


class Rosenbrock(object):

    """Rosenbrock loglikelihood as described in the PolyChord paper."""

    def __init__(self, a=1, b=100, nderived=0):
        """
        Define likelihood hyperparameter values.

        Parameters
        ----------
        theta: 1d numpy array
            Parameters.
        a: float, optional
        b: float, optional
        nderived: int, optional
            Number of derived parameters.
        """
        self.a = a
        self.b = b
        self.nderived = nderived

    def __call__(self, theta):
        """
        Calculate loglikelihood(theta), as well as any derived parameters.

        N.B. this loglikelihood is not normalised.

        Parameters
        ----------
        theta: float or 1d numpy array

        Returns
        -------
        logl: float
            Loglikelihood
        phi: list of length nderived
            Any derived parameters.
        """
        ftheta = 0
        for i in range(len(theta) - 1):
            ftheta += (self.a - theta[i]) ** 2
            ftheta += self.b * ((theta[i + 1] - (theta[i] ** 2)) ** 2)
        logl = -ftheta
        return logl, [0.0] * self.nderived


class GaussianMix(object):

    """
    Gaussian mixture likelihood in :math:`\\ge 2` dimensions with up to
    4 compoents.

    Each component has the same standard deviation :math:`\\sigma`, and
    they their centres respectively have :math:`(\\theta_1, \\theta_2)`
    coordinates:

    (0, sep), (0, -sep), (sep, 0), (-sep, 0).
    """

    def __init__(self, sep=4, weights=(0.4, 0.3, 0.2, 0.1), sigma=1,
                 nderived=0):
        """
        Define likelihood hyperparameter values.

        Parameters
        ----------
        sep: float
            Distance from each Gaussian to the origin.
        weights: iterable of floats
            weights of each Gaussian component.
        sigma: float
            Stanard deviation of Gaussian components.
        nderived: int, optional
            Number of derived parameters.
        """
        assert len(weights) in [2, 3, 4], (
            'Weights must have 2, 3 or 4 components. Weights=' + str(weights))
        assert np.isclose(sum(weights), 1), (
            'Weights must sum to 1! Weights=' + str(weights))
        self.nderived = nderived
        self.weights = weights
        self.sigmas = [sigma] * len(weights)
        positions = []
        positions.append(np.asarray([0, sep]))
        positions.append(np.asarray([0, -sep]))
        positions.append(np.asarray([sep, 0]))
        positions.append(np.asarray([-sep, 0]))
        self.positions = positions[:len(weights)]

    def __call__(self, theta):
        """
        Calculate loglikelihood(theta), as well as any derived parameters.

        N.B. this loglikelihood is not normalised.

        Parameters
        ----------
        theta: float or 1d numpy array

        Returns
        -------
        logl: float
            Loglikelihood
        phi: list of length nderived
            Any derived parameters.
        """
        thetas = []
        for pos in self.positions:
            thetas.append(copy.deepcopy(theta))
            thetas[-1][:2] -= pos
        logls = [(Gaussian(sigma=self.sigmas[i])(thetas[i])[0]
                  + np.log(self.weights[i])) for i in range(len(self.weights))]
        logl = scipy.special.logsumexp(logls)
        return logl, [0.0] * self.nderived


class LogGammaMix(object):

    """The loggamma mix used in Beaujean and Caldwell (2013) and Feroz et al.
    (2013)."""

    def __init__(self, nderived=0):
        """
        Define likelihood hyperparameter values.

        Parameters
        ----------
        nderived: int, optional
            Number of derived parameters.
        """
        self.nderived = nderived

    def __call__(self, theta):
        """
        Calculate loglikelihood(theta), as well as any derived parameters.

        N.B. this loglikelihood is not normalised.

        Parameters
        ----------
        theta: float or 1d numpy array

        Returns
        -------
        logl: float
            Loglikelihood
        phi: list of length nderived
            Any derived parameters.
        """
        assert theta.shape[0] % 2 == 0, (
            'ndim={} must be even'.format(theta.shape))
        # first component is
        # l(t) = 0.5 * loggamma(t - 10, 1, 1) + 0.5 * loggamma(t + 10, 1, 1)
        logl = np.log(0.5) + scipy.special.logsumexp(
            [log_loggamma_pdf(th, alpha=1, beta=1) for th in
             [theta[0] - 10, theta[0] + 10]])
        # second component is
        # l(t) = 0.5 * N(t - 10, 1, 1) + 0.5 * N(t + 10, 1, 1)
        logl += np.log(0.5) + scipy.special.logsumexp(
            [log_gaussian_pdf(th, sigma=1) for th in
             [theta[1] - 10, theta[1] + 10]])
        if theta.shape[0] > 2:
            # remaining components are half loggamma and half gaussian
            bound_ind = (theta.shape[0] // 2) + 1
            logl += log_loggamma_pdf(theta[2:bound_ind], alpha=1, beta=1)
            logl += log_gaussian_pdf(theta[bound_ind:], sigma=1)
        return logl, [0.0] * self.nderived


# Helper functions
# ----------------

def log_loggamma_pdf(theta, alpha=1, beta=1):
    """
    Log gamma distribution, with each component of theta independently having
    PDF:
    .. math::

        f(x) = \\frac{
            \\mathrm{e}^{\\beta x} \\mathrm{e}^{-\\mathrm{e}^x / \\alpha}
            }{\\alpha^{\\beta} \\Gamma(\\beta)}

    This function returns :math:`\\log f(x)`.

    Parameters
    ----------
    theta: float or array
        Where to evaluate :math:`\\log f(x)`. Values
        :math:`\\in (-\\inf, \\inf)`.
    alpha: float, > 0
        Scale parameter
    beta: float, > 0
        Shape parameter

    Returns
    -------
    logl: float
    """
    # log of numerator
    logl = beta * theta
    logl += -np.exp(theta) / alpha
    # log of denominator
    logl -= np.log(alpha) * beta
    logl -= scipy.special.gammaln(beta)
    # If there are many components, sum their log-space consributions
    if not isinstance(logl, (float, int)):
        assert logl.shape == theta.shape, (
            'logl={} theta={}'.format(logl, theta))
        logl = np.sum(logl)
    return logl


def log_gaussian_pdf(theta, sigma=1, mu=0, ndim=None):
    """Log of uncorrelated Gaussian pdf."""
    if ndim is None:
        try:
            ndim = len(theta)
        except TypeError:
            assert isinstance(theta, (float, int)), theta
            ndim = 1
    logl = -(np.sum((theta - mu) ** 2) / (2 * sigma ** 2))
    logl -= np.log(2 * np.pi * (sigma ** 2)) * ndim / 2.0
    return logl
