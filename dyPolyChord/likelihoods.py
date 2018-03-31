#!/usr/bin/env python
"""
Likelihood functions for use with PyPolyChord.
"""
import numpy as np
import scipy.special


def gaussian(theta, sigma=0.5, n_derived=0):
    """ Simple Gaussian Likelihood centered on the origin."""
    phi = [0.0] * n_derived
    dim = len(theta)
    rad2 = sum([t ** 2 for t in theta])
    logl = -np.log(2 * np.pi * (sigma ** 2)) * dim / 2.0
    logl += -rad2 / (2 * sigma ** 2)
    return logl, phi


def gaussian_mix(theta, n_derived=0, sep=4, weights=(0.4, 0.3, 0.2, 0.1)):
    """Gaussian mixture model."""
    assert len(weights) in [2, 3, 4], (
        'So far only set up for 2, 3 or 4 components. Weights=' + str(weights))
    assert np.isclose(sum(weights), 1), (
        'Weights must sum to 1! Weights=' + str(weights))
    sigmas = [1] * len(weights)
    positions = [(0, sep), (0, -sep), (sep, 0), (-sep, 0)][:len(weights)]
    thetas = []
    for pos in positions:
        thetas.append([theta[0] - pos[0], theta[1] - pos[1]] + theta[2:])
    logls = [(gaussian(thetas[i], sigmas[i])[0] + np.log(weights[i]))
             for i in range(len(weights))]
    logl = scipy.special.logsumexp(logls)
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
    dim = len(theta)
    ftheta = A * dim
    for th in theta:
        ftheta += (th ** 2) - A * np.cos(2 * np.pi * th)
    logl = -ftheta
    return logl, phi


def rosenbrock(theta, n_derived=0, a=1, b=100):
    """Rosenbrock Likelihood as described in the PolyChord paper."""
    phi = [0.0] * n_derived
    dim = len(theta)
    ftheta = 0
    for i in range(dim - 1):
        ftheta += (a - theta[i]) ** 2
        ftheta += b * ((theta[i + 1] - (theta[i] ** 2)) ** 2)
    logl = -ftheta
    return logl, phi
