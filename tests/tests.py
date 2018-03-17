#!/usr/bin/env python
"""
Test the dyPolyChord module.
"""
import os
import shutil
import unittest
import functools
import scipy.special
import numpy as np
# import numpy.testing
from PyPolyChord.settings import PolyChordSettings
# # import PyPolyChord
# import nestcheck.parallel_utils
import dypypolychord.likelihoods as likelihoods
import dypypolychord.priors as priors
import dypypolychord
# import dypypolychord.save_load_utils as slu
# # import mpi4py

TEST_DIR = 'test_chains'

SETTINGS_KWARGS = {
    'do_clustering': True,
    'posteriors': False,
    'equals': False,
    'write_dead': True,
    'read_resume': False,
    'write_resume': False,
    'write_stats': True,
    'write_prior': False,
    'write_live': False,
    'num_repeats': 1,
    'feedback': -1,
    'cluster_posteriors': False,
    'precision_criterion': 0.001,
    'seed': 1,
    'base_dir': TEST_DIR,
    'nlive': 10,
    'nlives': {}}


#class TestRun(unittest.TestCase):
#
#    def setUp(self):
#        """Check TEST_DIR does not already exist."""
#        assert not os.path.exists(TEST_DIR), \
#            ('Directory ' + TEST_DIR + ' exists! Tests use this ' +
#             'dir to check caching then delete it afterwards, so the path ' +
#             'should be left empty.')
#        self.ndims = 2
#        self.likelihood = functools.partial(likelihoods.gaussian, sigma=1)
#        self.prior = functools.partial(priors.gaussian, prior_scale=10)
#        self.settings = PolyChordSettings(self.ndims, 0, **SETTINGS_KWARGS)
#
#    # def tearDown(self):
#    #     """Remove any caches created by the tests."""
#    #     try:
#    #         shutil.rmtree(TEST_DIR)
#    #     except FileNotFoundError:
#    #         pass
#
#    def test_dynamic_evidence(self):
#        dynamic_goal = 0
#        dypypolychord.run_dypypolychord(
#            self.settings, self.likelihood, self.prior, self.ndims,
#            dynamic_goal=dynamic_goal, ninit=5, dyn_nlive_step=10)
#        run = dypypolychord.dynamic_processing.process_dypypolychord_run(
#            self.settings.base_dir + '/' + self.settings.file_root,
#            dynamic_goal)
#        print(run)
#        # dynamic settings - only used if dynamic_goal is not None


class TestPriors(unittest.TestCase):

    def test_uniform(self):
        """Check uniform prior."""
        prior_scale = 5
        hypercube = list(np.random.random(5))
        theta = priors.uniform(hypercube, prior_scale=prior_scale)
        for i, th in enumerate(theta):
            self.assertEqual(
                th, (hypercube[i] * 2 * prior_scale) - prior_scale)

    def test_gaussian(self):
        """Check spherically symmetric Gaussian prior centred on the origin."""
        prior_scale = 5
        hypercube = list(np.random.random(5))
        theta = priors.gaussian(hypercube, prior_scale=prior_scale)
        for i, th in enumerate(theta):
            self.assertAlmostEqual(
                th, (scipy.special.erfinv(hypercube[i] * 2 - 1) *
                     prior_scale * np.sqrt(2)), places=12)


class TestLikelihoods(unittest.TestCase):

    def test_gaussian(self):
        sigma = 1
        dim = 5
        theta = list(np.random.random(dim))
        logl, phi = likelihoods.gaussian(theta, sigma=sigma)
        self.assertAlmostEqual(
            logl, -(sum([th ** 2 for th in theta]) / (2 * sigma ** 2) +
                    np.log(2 * np.pi * sigma ** 2) * (dim / 2)), places=12)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)

    def test_gaussian_shell(self):
        dim = 5
        sigma = 1
        rshell = 2
        theta = list(np.random.random(dim))
        r = sum([th ** 2 for th in theta]) ** 0.5
        logl, phi = likelihoods.gaussian_shell(theta, sigma=sigma,
                                               rshell=rshell)
        self.assertAlmostEqual(
            logl, -((r - rshell) ** 2) / (2 * (sigma ** 2)), places=12)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)


if __name__ == '__main__':
    unittest.main()
