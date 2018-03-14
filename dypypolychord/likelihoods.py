#!/usr/bin/env python
"""
Likelihood functions for use with PyPolyChord.
"""
import numpy as np
import scipy.special


def gaussian(theta, sigma=0.5, n_derived=0):
    """ Simple Gaussian Likelihood centred on the origin."""
    phi = [0.0] * n_derived
    dim_theta = len(theta)
    rad2 = sum([t ** 2 for t in theta])
    if n_derived != 0:
        phi[0] = np.sqrt(rad2) * 0
    logl = -np.log(2 * np.pi * sigma * sigma) * dim_theta / 2.0
    logl += -rad2 / 2 / sigma / sigma
    return logl, phi


def twin_gaussian(theta, sigma=0.5, sep_sigma=10, n_derived=0):
    """Two Gaussians seperated by sep_sigma."""
    theta1 = [theta[0] + sigma * 0.5 * sep_sigma] + theta[1:]
    theta2 = [theta[0] - sigma * 0.5 * sep_sigma] + theta[1:]
    twin_logls = [gaussian(t, sigma)[0] for t in [theta1, theta2]]
    logl = scipy.special.logsumexp(twin_logls) - np.log(2)
    phi = [0.0] * n_derived
    return logl, phi


def gaussian_shell(theta, sigma=0.2, rshell=2, n_derived=0):
    """Gaussian Shell likelihood."""
    phi = [0.0] * n_derived
    rad = np.sqrt(sum([t ** 2 for t in theta]))
    logl = - ((rad - rshell) ** 2) / (2 * sigma ** 2)
    return logl, phi


def rastrigin(theta, n_derived=0, A=10):
    """Rastrigin Likelihood as described in the PolyChord paper"""
    phi = [0.0] * n_derived
    dim_theta = len(theta)
    ftheta = A * dim_theta
    for th in theta:
        ftheta += (th ** 2) - A * np.cos(2 * np.pi * th)
    logl = -ftheta
    return logl, phi


def rosenbrock(theta, n_derived=0, a=1, b=100):
    """Rosenbrock Likelihood as described in the PolyChord paper."""
    phi = [0.0] * n_derived
    dim_theta = len(theta)
    ftheta = 0
    for i in range(dim_theta - 1):
        ftheta += (a - theta[i]) ** 2
        ftheta += b * ((theta[i + 1] - (theta[i] ** 2)) ** 2)
    logl = -ftheta
    return logl, phi
