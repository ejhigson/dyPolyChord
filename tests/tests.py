#!/usr/bin/env python
"""
Test the dyPolyChord module.
"""
import copy
import os
import shutil
import unittest
import functools
import scipy.special
import numpy as np
import numpy.testing
import nestcheck.estimators as e
import nestcheck.dummy_data
import dyPolyChord
import dyPolyChord.python_likelihoods as likelihoods
import dyPolyChord.python_priors as priors
import dyPolyChord.output_processing
try:
    import dyPolyChord.pypolychord_utils
except ImportError:
    pass

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
    # Set precision_criterion low to avoid non-deterministic likelihood errors.
    # These occur due in the low dimension and low and nlive cases we use for
    # fast testing as runs sometimes get very close to the peak where the
    # likelihood becomes approximately constant.
    'precision_criterion': 0.01,
    'seed': 1,
    'max_ndead': -1,
    'base_dir': TEST_CACHE_DIR,
    'file_root': 'test_run',
    'nlive': 2,  # 50,  # used for nlive_const
    'nlives': {}}

def dummy_run_func(settings, ndim=2, ndead_term=10):
    nthread = settings['nlive']
    if settings['max_ndead'] <= 0:
        ndead = ndead_term
    else:
        ndead = min(ndead_term, settings['max_ndead'])
    if 'nlives' not in settings or not settings['nlives']:
        assert ndead % settings['nlive'] == 0, (
            'ndead={0}, nlive={1}'.format(ndead, settings['nlive']))
    nsample = ndead // settings['nlive']
    nsample += 1  # PolyChord includes remaining live points in each thread
    run = nestcheck.dummy_data.get_dummy_run(
        nthread, nsample, ndim)
    # if settings['read_resume']:
    #     run['logl'] += 0.5
    #     run['thread_min_max'][:, 1] += 0.5
    dead = nestcheck.dummy_data.run_dead_points_array(run)
    root = os.path.join(settings['base_dir'], settings['file_root'])
    np.savetxt(root + '_dead-birth.txt', dead)
    if 'write_resume' in settings:
        if 'write_resume':
            np.savetxt(root + '.resume', np.zeros(10))
    nestcheck.dummy_data.write_dummy_polychord_stats(
        settings['file_root'], settings['base_dir'], ndead=dead.shape[0])


class TestRunDyPolyChord(unittest.TestCase):

    def setUp(self):
        """Make a directory for saving test results."""
        # assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        if not os.path.exists(TEST_CACHE_DIR):
            os.makedirs(TEST_CACHE_DIR)
        self.ninit = 2
        self.nlive_const = 4
        self.run_func = dummy_run_func

#    def tearDown(self):
#        """Remove any caches saved by the tests."""
#        try:
#            shutil.rmtree(TEST_CACHE_DIR)
#        except FileNotFoundError:
#            pass

    def test_run_dypolychord_unexpected_kwargs(self):
        self.assertRaises(
            TypeError, dyPolyChord.run_dypolychord,
            lambda x: None, SETTINGS_KWARGS, 1,
            unexpected=1)

    def test_dynamic_evidence(self):
        dynamic_goal = 0
        seed_increment = 100
        dyPolyChord.run_dypolychord(
            self.run_func, SETTINGS_KWARGS, dynamic_goal,
            init_step=self.ninit, ninit=self.ninit, print_time=True,
            seed_increment=seed_increment,
            nlive_const=self.nlive_const)
        run = dyPolyChord.output_processing.process_dypolychord_run(
            SETTINGS_KWARGS['file_root'], SETTINGS_KWARGS['base_dir'],
            dynamic_goal=dynamic_goal)
        print(run)
        # self.assertEqual(run['logl'][0], -86.7906522578895,
        #                  msg=self.random_seed_msg)
        # self.assertEqual(e.count_samples(run), 470)
        # self.assertAlmostEqual(e.logz(run), -5.99813424487512, places=12)
        # self.assertAlmostEqual(e.param_mean(run), -0.011725372420821929,
        #                        places=12)

    def test_dynamic_param(self):
        dynamic_goal = 1
        dyPolyChord.run_dypolychord(
            self.run_func, SETTINGS_KWARGS, dynamic_goal,
            init_step=self.ninit, ninit=self.ninit, print_time=True,
            nlive_const=self.nlive_const)
        run = dyPolyChord.output_processing.process_dypolychord_run(
            SETTINGS_KWARGS['file_root'], SETTINGS_KWARGS['base_dir'],
            dynamic_goal=dynamic_goal)
        # self.assertEqual(run['logl'][0], -63.6696935969461,
        #                  msg=self.random_seed_msg)
        # self.assertEqual(run['output']['resume_ndead'], 40)
        # self.assertEqual(run['output']['resume_nlike'], 85)
        # self.assertAlmostEqual(e.logz(run), -6.150334026130597, places=12)
        # self.assertAlmostEqual(e.param_mean(run), 0.1054356767020377,
        #                        places=12)
        # test nlive allocation before tearDown removes the runs


    def test_nlive_allocate(self):
        dynamic_goal = 1
        self.run_func(SETTINGS_KWARGS)
        dyn_info = dyPolyChord.nlive_allocation.allocate(
            SETTINGS_KWARGS, self.ninit, self.nlive_const,
            dynamic_goal, smoothing_filter=None)
        numpy.testing.assert_array_equal(
            dyn_info['init_nlive_allocation'],
            dyn_info['init_nlive_allocation_unsmoothed'])
        # Check turning off filter
        self.assertRaises(
            TypeError, dyPolyChord.nlive_allocation.allocate,
            SETTINGS_KWARGS, self.ninit, 100, dynamic_goal,
            unexpected=1)
        # Check no points remaining message
        settings = copy.deepcopy(SETTINGS_KWARGS)
        settings['max_ndead'] = 1
        # Check unexpected kwargs
        self.assertRaises(
            AssertionError, dyPolyChord.nlive_allocation.allocate,
            settings, self.ninit, 100, dynamic_goal)

@unittest.skip("Seeding problems")
class TestRunDyPolyChordOld(unittest.TestCase):

    def setUp(self):
        """Make a directory for saving test results."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        self.ninit = 20
        ndims = 2
        self.run_func = dyPolyChord.pypolychord_utils.get_python_run_func(
            functools.partial(likelihoods.gaussian, sigma=1),
            functools.partial(priors.uniform, prior_scale=10), ndims=ndims)
        self.random_seed_msg = (
            'This test is not affected by most of dyPolyChord, so if it fails '
            'your PolyChord install\'s random seed number generator is '
            'probably different to the one used to set the expected values.')

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_dynamic_evidence(self):
        dynamic_goal = 0
        dyPolyChord.run_dypolychord(
            self.run_func, SETTINGS_KWARGS, dynamic_goal,
            init_step=self.ninit, ninit=self.ninit, print_time=True)
        run = dyPolyChord.output_processing.process_dypolychord_run(
            SETTINGS_KWARGS['file_root'], SETTINGS_KWARGS['base_dir'],
            dynamic_goal=dynamic_goal)
        self.assertEqual(run['logl'][0], -86.7906522578895,
                         msg=self.random_seed_msg)
        self.assertEqual(e.count_samples(run), 470)
        self.assertAlmostEqual(e.logz(run), -5.99813424487512, places=12)
        self.assertAlmostEqual(e.param_mean(run), -0.011725372420821929,
                               places=12)

    def test_dynamic_param(self):
        dynamic_goal = 1
        dyPolyChord.run_dypolychord(
            self.run_func, SETTINGS_KWARGS, dynamic_goal,
            init_step=self.ninit, ninit=self.ninit, print_time=True)
        run = dyPolyChord.output_processing.process_dypolychord_run(
            SETTINGS_KWARGS['file_root'], SETTINGS_KWARGS['base_dir'],
            dynamic_goal=dynamic_goal)
        self.assertEqual(run['logl'][0], -63.6696935969461,
                         msg=self.random_seed_msg)
        self.assertEqual(run['output']['resume_ndead'], 40)
        self.assertEqual(run['output']['resume_nlike'], 85)
        self.assertAlmostEqual(e.logz(run), -6.150334026130597, places=12)
        self.assertAlmostEqual(e.param_mean(run), 0.1054356767020377,
                               places=12)
        # test nlive allocation before tearDown removes the runs
        dyn_info = dyPolyChord.nlive_allocation.allocate(
            SETTINGS_KWARGS, self.ninit, SETTINGS_KWARGS['nlive'],
            dynamic_goal, smoothing_filter=None)
        numpy.testing.assert_array_equal(
            dyn_info['init_nlive_allocation'],
            dyn_info['init_nlive_allocation_unsmoothed'])
        # Check turning off filter
        self.assertRaises(
            TypeError, dyPolyChord.nlive_allocation.allocate,
            SETTINGS_KWARGS, self.ninit, 100, dynamic_goal,
            unexpected=1)
        # Check no points remaining message
        settings = copy.deepcopy(SETTINGS_KWARGS)
        settings['max_ndead'] = 1
        # Check unexpected kwargs
        self.assertRaises(
            AssertionError, dyPolyChord.nlive_allocation.allocate,
            settings, self.ninit, 100, dynamic_goal)


class TestOutputProcessing(unittest.TestCase):

    def test_settings_root(self):
        root = dyPolyChord.output_processing.settings_root(
            'gaussian', 'uniform', 2, prior_scale=1, dynamic_goal=1,
            nlive_const=1, ninit=1, nrepeats=1, init_step=1)
        self.assertIsInstance(root, str)

    def test_settings_root_unexpected_kwarg(self):
        self.assertRaises(
            TypeError, dyPolyChord.output_processing.settings_root,
            'gaussian', 'uniform', 2, prior_scale=1, dynamic_goal=1,
            nlive_const=1, ninit=1, nrepeats=1, init_step=1,
            unexpected=1)

    def test_process_dypolychord_run_unexpected_kwarg(self):
        self.assertRaises(
            TypeError, dyPolyChord.output_processing.process_dypolychord_run,
            'file_root', 'base_dir', dynamic_goal=1, unexpected=1)

    def test_combine_resumed_dyn_run(self):
        """
        Test combining resumed dynamic and initial runs and removing
        duplicate points using dummy ns runs.
        """
        init = {'logl': np.asarray([0, 1, 2, 3]),
                'thread_labels': np.asarray([0, 1, 0, 1])}
        dyn = {'logl': np.asarray([0, 1, 2, 4, 5, 6]),
               'thread_labels': np.asarray([0, 1, 0, 1, 0, 1])}
        for run in [init, dyn]:
            run['theta'] = np.random.random((run['logl'].shape[0], 2))
            run['nlive_array'] = np.zeros(run['logl'].shape[0]) + 2
            run['nlive_array'][-1] = 1
            run['thread_min_max'] = np.asarray(
                [[-np.inf, run['logl'][-2]], [-np.inf, run['logl'][-1]]])
        comb = dyPolyChord.output_processing.combine_resumed_dyn_run(
            init, dyn, 1)
        numpy.testing.assert_array_equal(
            comb['thread_labels'], np.asarray([0, 1, 0, 2, 1, 0, 1]))
        numpy.testing.assert_array_equal(
            comb['logl'], np.asarray([0., 1., 2., 3., 4., 5., 6.]))
        numpy.testing.assert_array_equal(
            comb['nlive_array'], np.asarray([2., 2., 3., 3., 2., 2., 1.]))


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
