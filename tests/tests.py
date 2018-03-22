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
from PyPolyChord.settings import PolyChordSettings
import nestcheck.estimators as e
import dyPolyChord.likelihoods as likelihoods
import dyPolyChord.priors as priors
import dyPolyChord.output_processing
import dyPolyChord

TEST_CACHE_DIR = 'chains_test'
TEST_DIR_EXISTS_MSG = ('Directory ' + TEST_CACHE_DIR + ' exists! Tests use '
                       'this dir to check caching then delete it afterwards, '
                       'so the path should be left empty.')
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
    'base_dir': TEST_CACHE_DIR,
    'nlive': 10,
    'nlives': {}}


class TestRunDyPolyChord(unittest.TestCase):

    def setUp(self):
        """Make a directory for saving test results."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        self.ndims = 2
        self.ninit = 5
        self.dyn_nlive_step = 2
        self.likelihood = functools.partial(likelihoods.gaussian, sigma=1)
        self.prior = functools.partial(priors.gaussian, prior_scale=10)
        self.settings = PolyChordSettings(self.ndims, 0, **SETTINGS_KWARGS)

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_dynamic_evidence(self):
        dynamic_goal = 0
        self.settings.file_root = 'test_' + str(dynamic_goal) + '_1'
        dyPolyChord.run_dypolychord(
            self.settings, self.likelihood, self.prior, self.ndims,
            dynamic_goal=dynamic_goal, ninit=self.ninit,
            dyn_nlive_step=self.dyn_nlive_step, print_time=True)
        run = dyPolyChord.output_processing.process_dypolychord_run(
            self.settings.file_root, self.settings.base_dir,
            dynamic_goal=dynamic_goal)
        self.assertEqual(run['output']['nlike'], 549)
        self.assertAlmostEqual(-6.428463223381643, e.logz(run), places=12)
        self.assertAlmostEqual(-0.40217150003234303, e.param_mean(run),
                               places=12)

    def test_dynamic_param(self):
        dynamic_goal = 1
        self.settings.file_root = 'test_dg' + str(dynamic_goal) + '_1'
        dyPolyChord.run_dypolychord(
            self.settings, self.likelihood, self.prior, self.ndims,
            dynamic_goal=dynamic_goal, ninit=self.ninit,
            dyn_nlive_step=self.dyn_nlive_step, print_time=True)
        run = dyPolyChord.output_processing.process_dypolychord_run(
            self.settings.file_root, self.settings.base_dir,
            dynamic_goal=dynamic_goal)
        self.assertEqual(run['output']['nlike'], 1112)
        self.assertEqual(run['output']['resume_ndead'], 20)
        self.assertEqual(run['output']['resume_nlike'], 62)
        self.assertAlmostEqual(-7.711683486952201, e.logz(run), places=12)
        self.assertAlmostEqual(0.13859338573369537, e.param_mean(run),
                               places=12)

    def test_run_dypolychord_unexpected_kwargs(self):
        self.assertRaises(
            TypeError, dyPolyChord.run_dypolychord,
            self.settings, self.likelihood, self.prior, self.ndims,
            dynamic_goal=1, ninit=self.ninit,
            dyn_nlive_step=self.dyn_nlive_step, unexpected=1)
        self.assertRaises(
            TypeError, dyPolyChord.run_dypolychord,
            self.settings, self.likelihood, self.prior, self.ndims,
            dynamic_goal=0, ninit=self.ninit,
            dyn_nlive_step=self.dyn_nlive_step, unexpected=1)


class TestOutputProcessing(unittest.TestCase):

    def test_settings_root(self):
        root = dyPolyChord.output_processing.settings_root(
            'gaussian', 'uniform', 2, prior_scale=1, dynamic_goal=1,
            nlive_const=1, ninit=1, nrepeats=1, dyn_nlive_step=1, init_step=1)
        self.assertIsInstance(root, str)

    def test_settings_root_unexpected_kwarg(self):
        self.assertRaises(
            TypeError, dyPolyChord.output_processing.settings_root,
            'gaussian', 'uniform', 2, prior_scale=1, dynamic_goal=1,
            nlive_const=1, ninit=1, nrepeats=1, dyn_nlive_step=1, init_step=1,
            unexpected=1)

    def test_process_dypolychord_run_unexpected_kwarg(self):
        self.assertRaises(
            TypeError, dyPolyChord.output_processing.process_dypolychord_run,
            'file_root', 'base_dir', dynamic_goal=1, unexpected=1)


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

    def test_gaussian_mix(self):
        dim = 5
        theta = list(np.random.random(dim))
        _, phi = likelihoods.gaussian_mix(theta)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)

    def test_rastrigin(self):
        dim = 2
        theta = [0.] * dim
        logl, phi = likelihoods.rastrigin(theta)
        self.assertEqual(logl, 0)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)

    def test_rosenbrock(self):
        dim = 2
        theta = [0.] * dim
        logl, phi = likelihoods.rosenbrock(theta)
        self.assertAlmostEqual(logl, -1, places=12)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)


if __name__ == '__main__':
    unittest.main()
