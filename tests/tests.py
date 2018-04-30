#!/usr/bin/env python
"""
Test the dyPolyChord module.
"""
import copy
import os
import shutil
import unittest
import importlib
import functools
import scipy.special
import numpy as np
import numpy.testing
import nestcheck.estimators as e
import nestcheck.dummy_data
import dyPolyChord.python_likelihoods as likelihoods
import dyPolyChord.python_priors as priors
import dyPolyChord.output_processing
import dyPolyChord.polychord_utils
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
    # Set precision_criterion low to avoid non-deterministic like errors.
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


@unittest.skipIf(importlib.util.find_spec("PyPolyChord") is None,
                 'PyPolyChord not installed.')
class TestPyPolyChordUtils(unittest.TestCase):

    def test_python_run_func(self):
        import dyPolyChord.pypolychord_utils as pypolychord_utils
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        os.makedirs(TEST_CACHE_DIR)
        func = pypolychord_utils.get_python_run_func(
            likelihoods.gaussian, priors.uniform, 2)
        self.assertIsInstance(func, functools.partial)
        self.assertEqual(set(func.keywords.keys()),
                         {'nderived', 'ndim', 'likelihood', 'prior'})
        func({'base_dir': TEST_CACHE_DIR, 'file_root': 'temp', 'nlive': 5,
              'max_ndead': 5, 'feedback': -1})
        shutil.rmtree(TEST_CACHE_DIR)

    def test_comm(self):
        """Test MPI comm."""
        import dyPolyChord.pypolychord_utils as pypolychord_utils
        self.assertRaises(
            AssertionError, pypolychord_utils.python_run_func,
            {}, likelihood=1, prior=2, ndim=3, comm=DummyMPIComm(0))
        self.assertRaises(
            AssertionError, pypolychord_utils.python_run_func,
            {}, likelihood=1, prior=2, ndim=3, comm=DummyMPIComm(1))


class TestPolyChordUtils(unittest.TestCase):

    def setUp(self):
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        os.makedirs(TEST_CACHE_DIR)

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_format_settings(self):
        self.assertEqual(
            'T', dyPolyChord.polychord_utils.format_setting(True))
        self.assertEqual(
            'F', dyPolyChord.polychord_utils.format_setting(False))
        self.assertEqual(
            '1', dyPolyChord.polychord_utils.format_setting(1))
        self.assertEqual(
            '1 2', dyPolyChord.polychord_utils.format_setting([1, 2]))

    def test_get_prior_block_str(self):
        name = 'uniform'
        prior_params = [1, 2]
        expected = ('P : p{0} | \\theta_{{{0}}} | {1} | {2} | {3} |'
                    .format(1, 1, name, 1))
        expected += dyPolyChord.polychord_utils.format_setting(prior_params)
        expected += '\n'
        self.assertEqual(dyPolyChord.polychord_utils.get_prior_block_str(
            name, prior_params, 1, speed=1, block=1), expected)

    def test_get_prior_block_unexpected_kwargs(self):
        self.assertRaises(
            TypeError, dyPolyChord.polychord_utils.get_prior_block_str,
            'param_name', (1, 2), 2, unexpected=1)

    def test_write_ini(self):
        settings = {'nlive': 50, 'nlives': {-20.0: 100, -10.0: 200}}
        prior_block_str = 'prior_block\n'
        derived_str = 'derived'
        file_path = os.path.join(TEST_CACHE_DIR, 'temp.ini')
        dyPolyChord.polychord_utils.write_ini(
            settings, prior_block_str, file_path, derived_str=derived_str)
        with open(file_path, 'r') as ini_file:
            lines = ini_file.readlines()
        self.assertEqual(lines[-2], prior_block_str)
        self.assertEqual(lines[-1], derived_str)
        # Use sorted as ini lines written from dict.items() so order not
        # guarenteed.
        self.assertEqual(sorted(lines[:3]),
                         ['loglikes = -20.0 -10.0\n',
                          'nlive = 50\n',
                          'nlives = 100 200\n'])

    def test_compiled_run_func(self):
        func = dyPolyChord.polychord_utils.get_compiled_run_func(
            'echo', 'this is a dummy prior block string')
        self.assertIsInstance(func, functools.partial)
        self.assertEqual(set(func.keywords.keys()),
                         {'derived_str', 'ex_path', 'prior_block_str'})
        func({'base_dir': TEST_CACHE_DIR, 'file_root': 'temp'})


class TestRunDyPolyChord(unittest.TestCase):

    def setUp(self):
        """Set up function for saving dummy ns runs, a directory for saving
        test results and some settings."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        os.makedirs(TEST_CACHE_DIR)
        self.random_seed_msg = (
            'This test is not affected by dyPolyChord, so if it fails '
            'your numpy random seed number generator is '
            'probably different to the one used to set the expected values.')
        self.ninit = 2
        self.nlive_const = 4
        self.run_func = functools.partial(
            dummy_run_func, ndim=2, ndead_term=10, seed=1, logl_range=10)
        self.settings = {'base_dir': TEST_CACHE_DIR,
                         'file_root': 'test_run',
                         'seed': 1,
                         'nlives': {},
                         'write_resume': False,
                         'max_ndead': -1}

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_run_dypolychord_unexpected_kwargs(self):
        self.assertRaises(
            TypeError, dyPolyChord.run_dypolychord,
            lambda x: None, 1, {}, unexpected=1)

    def test_dynamic_evidence(self):
        dynamic_goal = 0
        self.settings['max_ndead'] = 24
        dyPolyChord.run_dypolychord(
            self.run_func, dynamic_goal, self.settings,
            init_step=self.ninit, ninit=self.ninit,
            nlive_const=self.nlive_const)
        with self.assertWarns(UserWarning):
            run = dyPolyChord.output_processing.process_dypolychord_run(
                self.settings['file_root'], self.settings['base_dir'],
                dynamic_goal=dynamic_goal, logl_warn_only=True)
        self.assertAlmostEqual(run['logl'][0], 0.0011437481734488664,
                               msg=self.random_seed_msg, places=12)
        self.assertEqual(e.count_samples(run), 24)
        self.assertAlmostEqual(e.logz(run), 5.130048204496198, places=12)

    def test_dynamic_param(self):
        dynamic_goal = 1
        dyPolyChord.run_dypolychord(
            self.run_func, dynamic_goal, self.settings,
            init_step=self.ninit, ninit=self.ninit,
            nlive_const=self.nlive_const)
        with self.assertWarns(UserWarning):
            run = dyPolyChord.output_processing.process_dypolychord_run(
                self.settings['file_root'], self.settings['base_dir'],
                dynamic_goal=dynamic_goal, logl_warn_only=True)
        # test nlive allocation before tearDown removes the runs
        self.assertAlmostEqual(run['logl'][0], 0.0011437481734488664,
                               msg=self.random_seed_msg, places=12)
        self.assertEqual(e.count_samples(run), 16)
        self.assertAlmostEqual(e.logz(run), 4.170019624479282, places=12)
        self.assertEqual(run['output']['resume_ndead'], 6)

    def test_comm(self):
        """Test MPI comm."""
        dynamic_goal = 1
        self.assertRaises(
            AssertionError, dyPolyChord.run_dypolychord,
            self.run_func, dynamic_goal, self.settings,
            init_step=self.ninit, ninit=self.ninit,
            nlive_const=self.nlive_const, comm=DummyMPIComm(0))


class TestNliveAllocation(unittest.TestCase):

    def test_allocate(self):
        dynamic_goal = 1
        run = nestcheck.dummy_data.get_dummy_run(2, 10, ndim=2, seed=0)
        with self.assertWarns(UserWarning):
            dyn_info = dyPolyChord.nlive_allocation.allocate(
                run, 40, dynamic_goal, smoothing_filter=None)
        numpy.testing.assert_array_equal(
            dyn_info['init_nlive_allocation'],
            dyn_info['init_nlive_allocation_unsmoothed'])
        # Check no points remaining error
        self.assertRaises(
            AssertionError, dyPolyChord.nlive_allocation.allocate,
            run, 1, dynamic_goal)

    def test_dyn_nlive_array_warning(self):
        dynamic_goal = 0
        run = nestcheck.dummy_data.get_dummy_run(2, 10, ndim=2, seed=0)
        smoothing = (lambda x: (x + 100 * np.asarray(list(range(x.shape[0])))))
        with self.assertWarns(UserWarning):
            dyn_info = dyPolyChord.nlive_allocation.allocate(
                run, 40, dynamic_goal, smoothing_filter=smoothing)
        numpy.testing.assert_array_equal(
            dyn_info['init_nlive_allocation'],
            dyn_info['init_nlive_allocation_unsmoothed'])

    def test_sample_importance(self):
        """Check sample importance provides expected results."""
        run = nestcheck.dummy_data.get_dummy_thread(
            4, ndim=2, seed=0, logl_range=1)
        imp = dyPolyChord.nlive_allocation.sample_importance(run, 0.5)
        numpy.testing.assert_allclose(
            np.asarray([0.66121679, 0.23896365, 0.08104094, 0.01877862]),
            imp)


@unittest.skip("Seeding problems")
class TestRunDyPolyChordOld(unittest.TestCase):

    def setUp(self):
        """Make a directory for saving test results."""
        assert not os.path.exists(TEST_CACHE_DIR), TEST_DIR_EXISTS_MSG
        self.ninit = 20
        ndim = 2
        self.run_func = dyPolyChord.pypolychord_utils.get_python_run_func(
            functools.partial(likelihoods.gaussian, sigma=1),
            functools.partial(priors.uniform, prior_scale=10), ndim=ndim)
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
            self.run_func, dynamic_goal, settings_dict=SETTINGS_KWARGS,
            init_step=self.ninit, ninit=self.ninit)
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
            self.run_func, dynamic_goal, settings_dict=SETTINGS_KWARGS,
            init_step=self.ninit, ninit=self.ninit)
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


# Helper functions
# ----------------


def dummy_run_func(settings, **kwargs):
    """
    Produces dummy PolyChord output files for use in testing.
    """
    ndim = kwargs.pop('ndim', 2)
    ndead_term = kwargs.pop('ndead_term', 10)
    seed = kwargs.pop('seed', 1)
    logl_range = kwargs.pop('logl_range', 10)
    kwargs.pop('comm', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    nthread = settings['nlive']
    if settings['max_ndead'] <= 0:
        ndead = ndead_term
    else:
        ndead = min(ndead_term, settings['max_ndead'])
    if 'nlives' not in settings or not settings['nlives']:
        assert ndead % nthread == 0, (
            'ndead={0}, nthread={1}'.format(ndead, nthread))
    nsample = ndead // nthread
    nsample += 1  # mimic PolyChord, which includes live point at termination
    # make dead points array
    run = nestcheck.dummy_data.get_dummy_run(
        nthread, nsample, seed=seed, logl_range=logl_range)
    nestcheck.ns_run_utils.get_run_threads(run)
    dead = nestcheck.dummy_data.run_dead_points_array(run)
    root = os.path.join(settings['base_dir'], settings['file_root'])
    np.savetxt(root + '_dead-birth.txt', dead)
    if settings['write_resume']:
        np.savetxt(root + '.resume', np.zeros(10))
    nestcheck.dummy_data.write_dummy_polychord_stats(
        settings['file_root'], settings['base_dir'], ndead=dead.shape[0])


class DummyMPIComm(object):
    """A dummy MPI.COMM object."""

    def __init__(self, rank):
        self.rank = rank

    def Get_rank(self):
        return self.rank

    def bcast(self, obj, root=0):
        raise AssertionError


if __name__ == '__main__':
    unittest.main()
