#!/usr/bin/env python
"""
Test suite for the dyPolyChord package.
"""
import os
import shutil
import unittest
import functools
import warnings
import scipy.special
import numpy as np
import numpy.testing
import nestcheck.estimators as e
import nestcheck.dummy_data
import nestcheck.write_polychord_output
import nestcheck.data_processing
import dyPolyChord.python_likelihoods as likelihoods
import dyPolyChord.python_priors as priors
import dyPolyChord.output_processing
import dyPolyChord.polychord_utils
import dyPolyChord.run_dynamic_ns
import dyPolyChord
try:
    # Only pypolychord_utils tests if PyPolyChord is installed
    import PyPolyChord
    import dyPolyChord.pypolychord_utils
    PYPOLYCHORD_AVAIL = True
except ImportError:
    PYPOLYCHORD_AVAIL = False


# Define a directory to output files produced by tests (this will be deleted
# when the tests finish).
TEST_CACHE_DIR = 'temp_test_data_to_delete'


def setUpModule():
    """Before running the test suite, check that TEST_CACHE_DIR does not
    already exist - as the tests will delete it."""
    assert not os.path.exists(TEST_CACHE_DIR), (
        'Directory ' + TEST_CACHE_DIR + ' exists! Tests use this directory to '
        'check caching then delete it afterwards, so its path should be left '
        'empty. You should manually delete or move ' + TEST_CACHE_DIR
        + ' before running the tests.')


@unittest.skipIf(not PYPOLYCHORD_AVAIL, 'PyPolyChord not installed.')
class TestRunDyPolyChordNumers(unittest.TestCase):

    """Tests for run_dypolychord which use PyPolyChord and check numerical
    outputs by setting random seed."""

    def setUp(self):
        """Make a directory for saving test results."""
        try:
            os.makedirs(TEST_CACHE_DIR)
        except FileExistsError:
            pass
        self.ninit = 20
        ndim = 2
        self.run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
            likelihoods.Gaussian(sigma=1), priors.Uniform(-10, 10), ndim=ndim)
        self.random_seed_msg = (
            'First dead point logl is {0} != {1}. '
            'This test is not affected by most of dyPolyChord, so if it fails '
            'your PolyChord install\'s random seed number generator is '
            'probably different to the one used to set the expected values.')
        self.settings = {
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
            # Set precision_criterion low to avoid non-deterministic like
            # errors. These occur due in the low dimension and low and nlive
            # cases we use for fast testing as runs sometimes get very close
            # to the peak where the likelihood becomes approximately constant.
            'precision_criterion': 0.01,
            'seed': 1,
            'max_ndead': -1,
            'base_dir': TEST_CACHE_DIR,
            'file_root': 'test_run',
            'nlive': 50,  # used for nlive_const
            'nlives': {}}

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except FileNotFoundError:
            pass

    def test_dynamic_evidence(self):
        """Test numerical results for nested sampling with dynamic_goal=0."""
        dynamic_goal = 0
        dyPolyChord.run_dypolychord(
            self.run_func, dynamic_goal, self.settings,
            init_step=self.ninit, ninit=self.ninit)
        run = nestcheck.data_processing.process_polychord_run(
            self.settings['file_root'], self.settings['base_dir'])
        first_logl = -89.9267531982664
        if not np.isclose(run['logl'][0], first_logl):
            warnings.warn(
                self.random_seed_msg.format(run['logl'][0], first_logl),
                UserWarning)
        else:
            self.assertEqual(e.count_samples(run), 548)
            self.assertAlmostEqual(e.param_mean(run), 0.0952046545193311,
                                   places=12)

    def test_dynamic_both_evidence_and_param(self):
        """Test numerical results for nested sampling with
        dynamic_goal=0.25."""
        dynamic_goal = 0.25
        dyPolyChord.run_dypolychord(
            self.run_func, dynamic_goal, self.settings,
            init_step=self.ninit, ninit=self.ninit)
        run = nestcheck.data_processing.process_polychord_run(
            self.settings['file_root'], self.settings['base_dir'])
        first_logl = -82.3731123424932
        if not np.isclose(run['logl'][0], first_logl):
            warnings.warn(
                self.random_seed_msg.format(run['logl'][0], first_logl),
                UserWarning)
        else:
            self.assertEqual(e.count_samples(run), 518)
            self.assertAlmostEqual(e.param_mean(run), 0.08488421272724601,
                                   places=12)

    def test_dynamic_param(self):
        """Test numerical results for nested sampling with dynamic_goal=1."""
        dynamic_goal = 1
        dyPolyChord.run_dypolychord(
            self.run_func, dynamic_goal, self.settings,
            init_step=self.ninit, ninit=self.ninit)
        run = nestcheck.data_processing.process_polychord_run(
            self.settings['file_root'], self.settings['base_dir'])
        first_logl = -73.2283115991452
        if not np.isclose(run['logl'][0], first_logl):
            warnings.warn(
                self.random_seed_msg.format(run['logl'][0], first_logl),
                UserWarning)
        else:
            self.assertAlmostEqual(e.param_mean(run), 0.21352566194422262,
                                   places=12)


class TestRunDynamicNS(unittest.TestCase):

    """Tests for the run_dynamic_ns.py module."""

    def setUp(self):
        """Set up function for make dummy PolyChord data, a directory for
        saving test results and some settings."""
        try:
            os.makedirs(TEST_CACHE_DIR)
        except FileExistsError:
            pass
        self.random_seed_msg = (
            'This test is not affected by dyPolyChord, so if it fails '
            'your numpy random seed number generator is probably different '
            'to the one used to set the expected values.')
        self.ninit = 2
        self.nlive_const = 4
        self.run_func = functools.partial(
            dummy_run_func, ndim=2, ndead_term=10, seed=1, logl_range=10)
        self.settings = {'base_dir': TEST_CACHE_DIR,
                         'file_root': 'test_run',
                         'seed': 1,
                         'max_ndead': -1,
                         'posteriors': True}

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except IOError:
            pass

    def test_run_dypolychord_unexpected_kwargs(self):
        """Check appropriate error is raised when an unexpected keyword
        argument is given."""
        self.assertRaises(
            TypeError, dyPolyChord.run_dypolychord,
            lambda x: None, 1, {}, unexpected=1)

    def test_dynamic_evidence(self):
        """Check run_dypolychord targeting evidence. This uses dummy
        PolyChord-format data."""
        dynamic_goal = 0
        self.settings['max_ndead'] = 24
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            dyPolyChord.run_dypolychord(
                self.run_func, dynamic_goal, self.settings,
                init_step=self.ninit, ninit=self.ninit,
                nlive_const=self.nlive_const, logl_warn_only=True,
                stats_means_errs=False)
            self.assertEqual(len(war), 3)
        # Check the mean value using the posteriors file (its hard to make a
        # dummy run_func which is realistic enough to not fail checks if we try
        # loading the output normally with
        # nesthcheck.data_processing.process_polychord_run).
        posteriors = np.loadtxt(os.path.join(
            self.settings['base_dir'], self.settings['file_root'] + '.txt'))
        # posteriors have columns: weight / max weight, -2*logl, [params]
        p1_mean = (np.sum(posteriors[:, 2] * posteriors[:, 0])
                   / np.sum(posteriors[:, 0]))
        self.assertAlmostEqual(p1_mean, 0.6509612992491138, places=12)

    def test_dynamic_param(self):
        """Check run_dypolychord targeting evidence. This uses dummy
        PolyChord-format data."""
        dynamic_goal = 1
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            dyPolyChord.run_dypolychord(
                self.run_func, dynamic_goal, self.settings,
                init_step=self.ninit, ninit=self.ninit,
                nlive_const=self.nlive_const, logl_warn_only=True,
                stats_means_errs=False)
            self.assertEqual(len(war), 2)
        # Check the mean value using the posteriors file (its hard to make a
        # dummy run_func which is realistic enough to not fail checks if we try
        # loading the output normally with
        # nesthcheck.data_processing.process_polychord_run).
        posteriors = np.loadtxt(os.path.join(
            self.settings['base_dir'], self.settings['file_root'] + '.txt'))
        # posteriors have columns: weight / max weight, -2*logl, [params]
        p1_mean = (np.sum(posteriors[:, 2] * posteriors[:, 0])
                   / np.sum(posteriors[:, 0]))
        self.assertAlmostEqual(p1_mean, 0.614126384660822, places=12)

    def test_comm(self):
        """Test run_dyPolyChord's comm argument, which is used for running
        python likelihoods using MPI parallelisation with mpi4py."""
        dynamic_goal = 1
        self.assertRaises(
            AssertionError, dyPolyChord.run_dypolychord,
            self.run_func, dynamic_goal, self.settings,
            init_step=self.ninit, ninit=self.ninit,
            nlive_const=self.nlive_const, comm=DummyMPIComm(0))

    def test_check_settings_dict(self):
        """Make sure settings are checked ok, including issuing warning if a
        setting with a mandatory value is given a different value."""
        settings = {'read_resume': True, 'equals': True, 'posteriors': False}
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            dyPolyChord.run_dynamic_ns.check_settings_dict(settings)
            self.assertEqual(len(war), 1)


class TestNliveAllocation(unittest.TestCase):

    """Tests for the nlive_allocation.py module."""

    def test_allocate(self):
        """Check the allocate function for computing where to put additional
        samples."""
        dynamic_goal = 1
        run = nestcheck.dummy_data.get_dummy_run(2, 10, ndim=2, seed=0)
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            dyn_info = dyPolyChord.nlive_allocation.allocate(
                run, 40, dynamic_goal, smoothing_filter=None)
            self.assertEqual(len(war), 1)
        numpy.testing.assert_array_equal(
            dyn_info['init_nlive_allocation'],
            dyn_info['init_nlive_allocation_unsmoothed'])
        # Check no points remaining error
        self.assertRaises(
            AssertionError, dyPolyChord.nlive_allocation.allocate,
            run, 1, dynamic_goal)

    def test_dyn_nlive_array_warning(self):
        """Check handling of case where nlive smoothing introduces unwanted
        convexity for dynamic_goal=0."""
        dynamic_goal = 0
        run = nestcheck.dummy_data.get_dummy_run(2, 10, ndim=2, seed=0)
        smoothing = (lambda x: (x + 100 * np.asarray(list(range(x.shape[0])))))
        with warnings.catch_warnings(record=True) as war:
            warnings.simplefilter("always")
            dyn_info = dyPolyChord.nlive_allocation.allocate(
                run, 40, dynamic_goal, smoothing_filter=smoothing)
            self.assertEqual(len(war), 1)
        numpy.testing.assert_array_equal(
            dyn_info['init_nlive_allocation'],
            dyn_info['init_nlive_allocation_unsmoothed'])

    def test_sample_importance(self):
        """Check sample importance provides expected results."""
        run = nestcheck.dummy_data.get_dummy_thread(
            4, ndim=2, seed=0, logl_range=1)
        imp = dyPolyChord.nlive_allocation.sample_importance(run, 0.5)
        self.assertEqual(run['logl'].shape, imp.shape)
        numpy.testing.assert_allclose(
            np.asarray([0.66121679, 0.23896365, 0.08104094, 0.01877862]),
            imp)


class TestOutputProcessing(unittest.TestCase):

    """Tests for the output_processing.py module."""

    def test_settings_root(self):
        """Check standard settings root string."""
        root = dyPolyChord.output_processing.settings_root(
            'gaussian', 'uniform', 2, prior_scale=1, dynamic_goal=1,
            nlive_const=1, ninit=1, nrepeats=1, init_step=1)
        self.assertEqual(
            'gaussian_uniform_1_dg1_1init_1is_2d_1nlive_1nrepeats', root)

    def test_settings_root_unexpected_kwarg(self):
        """Check appropriate error is raised when an unexpected keyword
        argument is given."""
        self.assertRaises(
            TypeError, dyPolyChord.output_processing.settings_root,
            'gaussian', 'uniform', 2, prior_scale=1, dynamic_goal=1,
            nlive_const=1, ninit=1, nrepeats=1, init_step=1,
            unexpected=1)

    def test_process_dypolychord_run_unexpected_kwarg(self):
        """Check appropriate error is raised when an unexpected keyword
        argument is given."""
        self.assertRaises(
            TypeError, dyPolyChord.output_processing.process_dypolychord_run,
            'file_root', 'base_dir', dynamic_goal=1, unexpected=1)

    def test_combine_resumed_dyn_run(self):
        """Test combining resumed dynamic and initial runs and removing
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
        self.assertEqual(set(comb.keys()),
                         {'nlive_array', 'theta', 'logl', 'thread_labels',
                          'thread_min_max'})
        numpy.testing.assert_array_equal(
            comb['thread_labels'], np.asarray([0, 1, 0, 2, 1, 0, 1]))
        numpy.testing.assert_array_equal(
            comb['logl'], np.asarray([0., 1., 2., 3., 4., 5., 6.]))
        numpy.testing.assert_array_equal(
            comb['nlive_array'], np.asarray([2., 2., 3., 3., 2., 2., 1.]))


class TestPolyChordUtils(unittest.TestCase):

    """Tests for the polychord_utils.py module."""

    def setUp(self):
        try:
            os.makedirs(TEST_CACHE_DIR)
        except FileExistsError:
            pass

    def tearDown(self):
        """Remove any caches saved by the tests."""
        try:
            shutil.rmtree(TEST_CACHE_DIR)
        except IOError:
            pass

    def test_format_settings(self):
        """Check putting settings dictionary values into the format needed for
        PolyChord .ini files."""
        self.assertEqual(
            'T', dyPolyChord.polychord_utils.format_setting(True))
        self.assertEqual(
            'F', dyPolyChord.polychord_utils.format_setting(False))
        self.assertEqual(
            '1', dyPolyChord.polychord_utils.format_setting(1))
        self.assertEqual(
            '1 2', dyPolyChord.polychord_utils.format_setting([1, 2]))

    def test_get_prior_block_str(self):
        """Check generating prior blocks in the format needed for PolyChord
        .ini files."""
        name = 'uniform'
        prior_params = [1, 2]
        expected = ('P : p{0} | \\theta_{{{0}}} | {1} | {2} | {3} |'
                    .format(1, 1, name, 1))
        expected += dyPolyChord.polychord_utils.format_setting(prior_params)
        expected += '\n'
        self.assertEqual(dyPolyChord.polychord_utils.get_prior_block_str(
            name, prior_params, 1, speed=1, block=1), expected)

    def test_get_prior_block_unexpected_kwargs(self):
        """Check appropriate error is raised when an unexpected keyword
        argument is given."""
        self.assertRaises(
            TypeError, dyPolyChord.polychord_utils.get_prior_block_str,
            'param_name', (1, 2), 2, unexpected=1)

    def test_write_ini(self):
        """Check writing a PolyChord .ini file from a dictionary of
        settings."""
        settings = {'nlive': 50, 'nlives': {-20.0: 100, -10.0: 200}}
        prior_str = 'prior_block\n'
        derived_str = 'derived'
        run_obj = dyPolyChord.polychord_utils.RunCompiledPolyChord(
            ':', prior_str, derived_str=derived_str)
        file_path = os.path.join(TEST_CACHE_DIR, 'temp.ini')
        run_obj.write_ini(settings, file_path)
        with open(file_path, 'r') as ini_file:
            lines = ini_file.readlines()
        self.assertEqual(lines[-2], prior_str)
        self.assertEqual(lines[-1], derived_str)
        # Use sorted as ini lines written from dict.items() so order not
        # guarenteed.
        self.assertEqual(sorted(lines[:3]),
                         ['loglikes = -20.0 -10.0\n',
                          'nlive = 50\n',
                          'nlives = 100 200\n'])

    def test_compiled_run_func(self):
        """
        Check function for running a compiled PolyChord likelihood from
        within python (via os.system).

        In place of an executable we just use the bash 'do nothing' command
        ':'.
        """
        func = dyPolyChord.polychord_utils.RunCompiledPolyChord(
            ':', 'this is a dummy prior block string')
        self.assertEqual(set(func.__dict__.keys()),
                         {'derived_str', 'ex_path', 'prior_str'})
        func({'base_dir': TEST_CACHE_DIR, 'file_root': 'temp'})


@unittest.skipIf(not PYPOLYCHORD_AVAIL, 'PyPolyChord not installed.')
class TestPyPolyChordUtils(unittest.TestCase):

    """
    Tests for the pypolychord_utils.py module.

    These are skipped if PyPolyChord is not installed as it is not needed for
    compiled likelihoods.
    """

    def test_python_run_func(self):
        """Check functions for running PolyChord via the PyPolyChord wrapper
        (as opposed to with a compiled likelihood) in the form needed for
        dynamic nested sampling."""
        try:
            os.makedirs(TEST_CACHE_DIR)
        except FileExistsError:
            pass
        func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
            likelihoods.Gaussian(), priors.Uniform(), 2)
        self.assertEqual(set(func.__dict__.keys()),
                         {'nderived', 'ndim', 'likelihood', 'prior'})
        func({'base_dir': TEST_CACHE_DIR, 'file_root': 'temp', 'nlive': 5,
              'max_ndead': 5, 'feedback': -1})
        shutil.rmtree(TEST_CACHE_DIR)

    def test_comm(self):
        """
        Test python_run_func's comm argument (used for MPI) has the expected
        behavior.
        """
        run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(1, 2, 3)
        self.assertRaises(
            AssertionError, run_func, {}, comm=DummyMPIComm(0))
        self.assertRaises(
            AssertionError, run_func, {}, comm=DummyMPIComm(1))


class TestPythonPriors(unittest.TestCase):

    """Tests for the python_priors.py module."""

    def test_uniform(self):
        """Check uniform prior."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta = priors.Uniform(-prior_scale, prior_scale)(hypercube)
        for i, th in enumerate(theta):
            self.assertEqual(
                th, (hypercube[i] * 2 * prior_scale) - prior_scale)

    def test_gaussian(self):
        """Check spherically symmetric Gaussian prior centred on the origin."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta = priors.Gaussian(prior_scale)(hypercube)
        for i, th in enumerate(theta):
            self.assertAlmostEqual(
                th, (scipy.special.erfinv(hypercube[i] * 2 - 1) *
                     prior_scale * np.sqrt(2)), places=12)


class TestPythonLikelihoods(unittest.TestCase):

    """Tests for the python_likelihoods.py module."""

    def test_gaussian(self):
        """Check the Gaussian likelihood."""
        sigma = 1
        dim = 5
        theta = np.random.random(dim)
        logl_expected = -(np.sum(theta ** 2) / (2 * sigma ** 2))
        logl_expected -= np.log(2 * np.pi * sigma ** 2) * (dim / 2.0)
        logl, phi = likelihoods.Gaussian(sigma=sigma)(theta)
        self.assertAlmostEqual(logl, logl_expected, places=12)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)

    def test_gaussian_shell(self):
        """Check the Gaussian shell likelihood."""
        dim = 5
        sigma = 1
        rshell = 2
        theta = np.random.random(dim)
        r = np.sum(theta ** 2) ** 0.5
        logl, phi = likelihoods.GaussianShell(
            sigma=sigma, rshell=rshell)(theta)
        self.assertAlmostEqual(
            logl, -((r - rshell) ** 2) / (2 * (sigma ** 2)), places=12)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)

    def test_gaussian_mix(self):
        """Check the Gaussian mixture model likelihood."""
        dim = 5
        theta = np.random.random(dim)
        _, phi = likelihoods.GaussianMix()(theta)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)

    def test_rastrigin(self):
        """Check the Rastrigin ("bunch of grapes") likelihood."""
        dim = 2
        theta = np.zeros(dim)
        logl, phi = likelihoods.Rastrigin()(theta)
        self.assertEqual(logl, 0)
        self.assertIsInstance(phi, list)
        self.assertEqual(len(phi), 0)

    def test_rosenbrock(self):
        """Check the Rosenbrock ("banana") likelihood."""
        dim = 2
        theta = np.zeros(dim)
        logl, phi = likelihoods.Rosenbrock()(theta)
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
        nthread, nsample, seed=seed, ndim=ndim, logl_range=logl_range)
    run['output'] = {'base_dir': settings['base_dir'],
                     'file_root': settings['file_root']}
    nestcheck.write_polychord_output.write_run_output(run)
    if settings['write_resume']:
        # if required, save a dummy resume file
        root = os.path.join(settings['base_dir'], settings['file_root'])
        np.savetxt(root + '.resume', np.zeros(10))


class DummyMPIComm(object):

    """A dummy mpi4py MPI.COMM object for testing."""

    def __init__(self, rank):
        self.rank = rank

    def Get_rank(self):
        """Dummy version of mpi4py MPI.COMM's Get_rank()."""
        return self.rank

    @staticmethod
    def bcast(_, root=0):
        """Dummy version of mpi4py MPI.COMM's bcast(data, root=0)
        method.
        AssertionError raising is used to allow behavior testing without
        running the whole of the run_dypolychord function in which the call is
        embedded."""
        if root == 0:
            raise AssertionError

if __name__ == '__main__':
    unittest.main()
