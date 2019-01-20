#!/usr/bin/env python
"""
Contains main function for running dynamic nested sampling.
"""
from __future__ import division  # Enforce float division in python2
import copy
import os
import traceback
import sys
import shutil
import warnings
import numpy as np
import scipy.signal
import nestcheck.data_processing
import nestcheck.io_utils
import nestcheck.write_polychord_output
import dyPolyChord.nlive_allocation
import dyPolyChord.output_processing
# pylint: disable=bare-except


__all__ = ['run_dypolychord', 'check_settings']


@nestcheck.io_utils.timing_decorator
def run_dypolychord(run_polychord, dynamic_goal, settings_dict_in, **kwargs):
    r"""Performs dynamic nested sampling using the algorithm described in
    Appendix F of "Dynamic nested sampling: an improved algorithm for
    parameter estimation and evidence calculation" (Higson et al., 2019).
    This proceeds in 4 steps:

    1) Generate an initial run with a constant number of live points
    :math:`n_\mathrm{init}`. This process is run in chuncks using PolyChord's
    max_ndead setting to allow periodic saving of .resume files so the initial
    run can be resumed at different points.

    2) Calculate an allocation of the number of live points at each likelihood
    for use in step 3. Also cleans up resume files and saves relevant
    information.

    3) Generate dynamic nested sampling run using the calculated live point
    allocation.

    4) Combine the initial and dynamic runs and write output files in the
    PolyChord format, and remove the intermediate output files produced.

    The output files are of the same format produced by ``PolyChord``, and
    contain posterior samples and an estimate of the Bayesian evidence.
    Further analysis, including estimating uncertainties, can be performed
    with ``nestcheck``.

    Like for ``PolyChord``, the output files are saved in base_dir (specified in
    settings_dict_in, default value is 'chains'). Their names are determined
    by file_root (also specified in settings_dict_in). ``dyPolyChord`` ensures
    the following following files are always produced:

        * [base_dir]/[file_root].stats: run statistics including an estimate of
          the Bayesian evidence;
        * [base_dir]/[file_root]_dead.txt: posterior samples;
        * [base_dir]/[file_root]_dead-birth.txt: as above but with an extra
          column containing information about when points were sampled.

    For more information about the output format, see ``PolyChord``'s
    documentation. Note that ``dyPolyChord`` is not able to produce all of the
    types of output files made by ``PolyChord`` - see check_settings's
    documentation for more information.
    In addition, a number of intermediate files are produced during the dynamic
    nested sampling process which are removed by default when the process
    finishes. See clean_extra_output's documentation for more details.

    Parameters
    ----------
    run_polychord: callable
        Callable which runs PolyChord with the desired likelihood and prior,
        and takes a settings dictionary as its argument.
    dynamic_goal: float or int
        Number in (0, 1) which determines how to allocate computational effort
        between parameter estimation and evidence calculation. See the dynamic
        nested sampling paper for more details.
    settings_dict: dict
        PolyChord settings to use (see check_settings for information on
        allowed and default settings).
    nlive_const: int, optional
        Used to calculate total number of samples if max_ndead not specified in
        settings. The total number of samples used is the estimated number that
        would be taken by a nested sampling run with a constant number of live
        points nlive_const.
    ninit: int, optional
        Number of live points to use for the initial exporatory run (Step 1).
    ninit_step: int, optional
        Number of samples taken between saving .resume files in Step 1.
    seed_increment: int, optional
        If seeding is used (PolyChord seed setting >= 0), this increment is
        added to PolyChord's random seed each time it is run to avoid
        repeated points.
        When running in parallel using MPI, PolyChord hashes the seed with the
        MPI rank using IEOR. Hence you need seed_increment to be > number of
        processors to ensure no two processes use the same seed.
        When running repeated results you need to increment the seed used for
        each run by some number >> seed_increment.
    smoothing_filter: func, optional
        Smoothing to apply to the nlive allocation (if any).
    stats_means_errs: bool, optional
        Whether to include estimates of logZ and parameter mean values and
        their uncertainties in the .stats file. This is passed to nestcheck's
        write_run_output; see its documentation for more details.
    clean: bool, optional
        Clean the additional output files made by dyPolyChord, leaving only
        output files for the combined run in PolyChord format.
        When debugging this can be set to False to allow inspection of
        intermediate output.
    """
    try:
        nlive_const = kwargs.pop('nlive_const', settings_dict_in['nlive'])
    except KeyError:
        nlive_const = kwargs.pop('nlive_const', 100)
    ninit = kwargs.pop('ninit', 10)
    init_step = kwargs.pop('init_step', ninit)
    seed_increment = kwargs.pop('seed_increment', 100)
    default_smoothing = (lambda x: scipy.signal.savgol_filter(
        x, 1 + (2 * ninit), 3, mode='nearest'))
    smoothing_filter = kwargs.pop('smoothing_filter', default_smoothing)
    comm = kwargs.pop('comm', None)
    stats_means_errs = kwargs.pop('stats_means_errs', True)
    clean = kwargs.pop('clean', True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Step 1: do initial run
    # ----------------------
    # set up rank if running with MPI
    if comm is not None:
        rank = comm.Get_rank()
    else:
        rank = 0
    settings_dict = None  # define for rank != 0
    if rank == 0:
        settings_dict_in, output_settings = check_settings(
            settings_dict_in)
        if (settings_dict_in['seed'] >= 0 and comm is not None and
                comm.Get_size() > 1):
            warnings.warn((
                'N.B. seeded results will not be reproducable when running '
                'dyPolyChord with multiple MPI processes. You have seed={} '
                'and {} MPI processes.').format(
                    settings_dict_in['seed'], comm.Get_size()), UserWarning)
        # Make a copy of settings dic so we dont edit settings
        settings_dict = copy.deepcopy(settings_dict_in)
        settings_dict['file_root'] = settings_dict['file_root'] + '_init'
        settings_dict['nlive'] = ninit
    if dynamic_goal == 0:
        # We definitely won't need to resume midway through in this case, so
        # just run PolyChod normally
        run_polychord(settings_dict, comm=comm)
        if rank == 0:
            final_seed = settings_dict['seed']
            if settings_dict['seed'] >= 0:
                final_seed += seed_increment
            step_ndead = None
            resume_outputs = None
    else:
        step_ndead, resume_outputs, final_seed = run_and_save_resumes(
            run_polychord, settings_dict, init_step, seed_increment, comm=comm)
    # Step 2: calculate an allocation of live points
    # ----------------------------------------------
    if rank == 0:
        try:
            # Get settings for dynamic run based on initial run
            settings_dict = process_initial_run(
                settings_dict_in, nlive_const=nlive_const,
                smoothing_filter=smoothing_filter,
                step_ndead=step_ndead, resume_outputs=resume_outputs,
                ninit=ninit, dynamic_goal=dynamic_goal,
                final_seed=final_seed)
        except:  # pragma: no cover
            if comm is None or comm.Get_size() == 1:
                raise
            else:
                # print error info
                traceback.print_exc(file=sys.stdout)
                print('Error in process with rank == 0: forcing MPI abort.')
                sys.stdout.flush()  # Make sure message prints before abort
                comm.Abort(1)
    # Step 3: do dynamic run
    # ----------------------
    run_polychord(settings_dict, comm=comm)
    # Step 4: process output and tidy
    # -------------------------------
    if rank == 0:
        try:
            # Combine initial and dynamic runs
            run = dyPolyChord.output_processing.process_dypolychord_run(
                settings_dict_in['file_root'], settings_dict_in['base_dir'],
                dynamic_goal=dynamic_goal)
            # Save combined output in PolyChord format
            nestcheck.write_polychord_output.write_run_output(
                run, stats_means_errs=stats_means_errs, **output_settings)
            if clean:
                # Remove temporary files
                root_name = os.path.join(settings_dict_in['base_dir'],
                                         settings_dict_in['file_root'])
                clean_extra_output(root_name)
        except:  # pragma: no cover
            if comm is None or comm.Get_size() == 1:
                raise
            else:
                # print error info
                traceback.print_exc(file=sys.stdout)
                print('Error in process with rank == 0: forcing MPI abort.')
                sys.stdout.flush()  # Make sure message prints before abort
                comm.Abort(1)


def process_initial_run(settings_dict_in, **kwargs):
    """Loads the initial exploratory run and analyses it to create the settings
    for the second, dynamic run.

    Parameters
    ----------
    settings_dict_in: dict
        Initial PolyChord settings (see check_settings for information on
        allowed and default settings).
    dynamic_goal: float or int
        Number in (0, 1) which determines how to allocate computational effort
        between parameter estimation and evidence calculation. See the dynamic
        nested sampling paper for more details.
    nlive_const: int
        Used to calculate total number of samples if max_ndead not specified in
        settings. The total number of samples used is the estimated number that
        would be taken by a nested sampling run with a constant number of live
        points nlive_const.
    ninit: int
        Number of live points to use for the initial exporatory run (Step 1).
    smoothing_filter: func
        Smoothing to apply to the nlive allocation (if any).
    step_ndead: list of ints
        Numbers of dead points at which resume files are saved.
    resume_outputs: dict
        Dictionary containing run output (contents of .stats file) at each
        resume. Keys are elements of step_ndead.
    final_seed: int
        Random seed at the end of the initial run.
    """
    dynamic_goal = kwargs.pop('dynamic_goal')
    nlive_const = kwargs.pop('nlive_const')
    ninit = kwargs.pop('ninit')
    smoothing_filter = kwargs.pop('smoothing_filter')
    step_ndead = kwargs.pop('step_ndead')
    resume_outputs = kwargs.pop('resume_outputs')
    final_seed = kwargs.pop('final_seed')
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    init_run = nestcheck.data_processing.process_polychord_run(
        settings_dict_in['file_root'] + '_init',
        settings_dict_in['base_dir'])
    # Calculate max number of samples
    if settings_dict_in['max_ndead'] > 0:
        samp_tot = settings_dict_in['max_ndead']
        assert (settings_dict_in['max_ndead']
                > init_run['logl'].shape[0]), (
                    'all points used in init run - '
                    'none left for dynamic')
    else:
        samp_tot = init_run['logl'].shape[0] * (nlive_const / ninit)
        assert nlive_const > ninit
    dyn_info = dyPolyChord.nlive_allocation.allocate(
        init_run, samp_tot, dynamic_goal,
        smoothing_filter=smoothing_filter)
    root_name = os.path.join(settings_dict_in['base_dir'],
                             settings_dict_in['file_root'])
    if dyn_info['peak_start_ind'] != 0:
        # subtract 1 as ndead=1 corresponds to point 0
        resume_steps = np.asarray(step_ndead) - 1
        # Work out which resume file to load
        resume_ndead = step_ndead[np.where(
            resume_steps < dyn_info['peak_start_ind'])[0][-1]]
        # copy resume step to dynamic file root
        shutil.copyfile(
            root_name + '_init_' + str(resume_ndead) + '.resume',
            root_name + '_dyn.resume')
        # Save resume info
        dyn_info['resume_ndead'] = resume_ndead
        try:
            dyn_info['resume_nlike'] = (
                resume_outputs[resume_ndead]['nlike'])
        except KeyError:
            pass  # protect from error reading nlike from .stats file
    nestcheck.io_utils.pickle_save(
        dyn_info, root_name + '_dyn_info', overwrite_existing=True)
    if dynamic_goal != 0:
        # Remove all the temporary resume files. Use set to avoid
        # duplicates as these cause OSErrors.
        for snd in set(step_ndead):
            os.remove(root_name + '_init_' + str(snd) + '.resume')
    settings_dict = copy.deepcopy(settings_dict_in)
    settings_dict['seed'] = final_seed
    if settings_dict['seed'] >= 0:
        assert settings_dict_in['seed'] >= 0, (
            'if input seed was <0 it should not have been edited')
    if dyn_info['peak_start_ind'] != 0:
        settings_dict['nlive'] = ninit
    else:
        settings_dict['nlive'] = dyn_info['nlives_dict'][
            min(dyn_info['nlives_dict'].keys())]
    settings_dict['nlives'] = dyn_info['nlives_dict']
    # To write .ini files correctly, read_resume must be type bool not
    # np.bool
    settings_dict['read_resume'] = (
        bool(dyn_info['peak_start_ind'] != 0))
    settings_dict['file_root'] = settings_dict_in['file_root'] + '_dyn'
    return settings_dict


# Helper functions
# ----------------

def clean_extra_output(root_name):
    """Clean the additional output files made by dyPolyChord, leaving only
    output files for the combined run in PolyChord format.

    Parameters
    ----------
    root_name: str
        File root. Equivalent to os.path.join(base_dir, file_root).
    """
    os.remove(root_name + '_dyn_info.pkl')
    for extra in ['init', 'dyn']:
        os.remove(root_name + '_{0}.stats'.format(extra))
        os.remove(root_name + '_{0}_dead-birth.txt'.format(extra))
        os.remove(root_name + '_{0}_dead.txt'.format(extra))
        # tidy up remaining .resume files (if the function has reach this
        # point, both the initial and dynamic runs have finished so we
        # shouldn't need to resume)
        try:
            os.remove(root_name + '_{0}.resume'.format(extra))
        except OSError:
            pass


def check_settings(settings_dict_in):
    """
    Checks the input dictionary of PolyChord settings. Issues warnings where
    these are not appropriate, and adds default values.

    Parameters
    ----------
    settings_dict_in: dict
        PolyChord settings to use.

    Returns
    -------
    settings_dict: dict
        Updated settings dictionary including default and mandatory values.
    output_settings: dict
        Settings for writing output files which are saved until the final
        output files are calculated at the end.
    """
    default_settings = {'nlive': 100,
                        'num_repeats': 20,
                        'file_root': 'temp',
                        'base_dir': 'chains',
                        'seed': -1,
                        'do_clustering': True,
                        'max_ndead': -1,
                        'equals': True,
                        'posteriors': True}
    mandatory_settings = {'nlives': {},
                          'write_dead': True,
                          'write_stats': True,
                          'write_paramnames': False,
                          'write_prior': False,
                          'write_live': False,
                          'write_resume': False,
                          'read_resume': False,
                          'cluster_posteriors': False,
                          'boost_posterior': 0.0}
    settings_dict = copy.deepcopy(settings_dict_in)
    # assign default settings
    for key, value in default_settings.items():
        if key not in settings_dict:
            settings_dict[key] = value
    # Produce warning if settings_dict_in has different values for any
    # mandatory settings.
    for key, value in mandatory_settings.items():
        if key in settings_dict_in and settings_dict_in[key] != value:
            warnings.warn((
                'dyPolyChord currently only allows the setting {0}={1}, '
                'so I am proceeding with this. You tried to specify {0}={2}.'
                .format(key, value, settings_dict_in[key])), UserWarning)
        settings_dict[key] = value
    # Extract output settings (not needed until later)
    output_settings = {}
    for key in ['posteriors', 'equals']:
        output_settings[key] = settings_dict[key]
        settings_dict[key] = False
    return settings_dict, output_settings


def run_and_save_resumes(run_polychord, settings_dict_in, init_step,
                         seed_increment, comm=None):
    """
    Run PolyChord pausing after every init_step dead points to save a resume
    file.

    Parameters
    ----------
    run_polychord: callable
        Callable which runs PolyChord with the desired likelihood and prior,
        and takes a settings dictionary as its argument.
    settings_dict: dict
        PolyChord settings to use (see check_settings for information on
        allowed and default settings).
    ninit_step: int, optional
        Number of samples taken between saving .resume files in Step 1.
    seed_increment: int, optional
        If seeding is used (PolyChord seed setting >= 0), this increment is
        added to PolyChord's random seed each time it is run to avoid
        repeated points.
        When running in parallel using MPI, PolyChord hashes the seed with the
        MPI rank using IEOR. Hence you need seed_increment to be > number of
        processors to ensure no two processes use the same seed.
        When running repeated results you need to increment the seed used for
        each run by some number >> seed_increment.
    comm: None or mpi4py MPI.COMM object, optional
        For MPI parallelisation.

    Returns
    -------
    step_ndead: list of ints
        Numbers of dead points at which resume files are saved.
    resume_outputs: dict
        Dictionary containing run output (contents of .stats file) at each
        resume. Keys are elements of step_ndead.
    final_seed: int
        Random seed. This is incremented after each run so it can be used
        when resuming without generating correlated points.
    """
    settings_dict = copy.deepcopy(settings_dict_in)
    # set up rank if running with MPI
    if comm is not None:
        # Define variables for rank != 0
        step_ndead = None
        resume_outputs = None
        final_seed = None
        # Get rank
        rank = comm.Get_rank()
    else:
        rank = 0
    if rank == 0:
        root_name = os.path.join(settings_dict['base_dir'],
                                 settings_dict['file_root'])
        try:
            os.remove(root_name + '.resume')
        except OSError:
            pass
        settings_dict['write_resume'] = True
        settings_dict['read_resume'] = True
        step_ndead = []
        resume_outputs = {}
    add_points = True
    while add_points:
        if rank == 0:
            settings_dict['max_ndead'] = (len(step_ndead) + 1) * init_step
        run_polychord(settings_dict, comm=comm)
        if rank == 0:
            try:
                if settings_dict['seed'] >= 0:
                    settings_dict['seed'] += seed_increment
                run_output = nestcheck.data_processing.process_polychord_stats(
                    settings_dict['file_root'], settings_dict['base_dir'])
                # Store run outputs for getting number of likelihood calls
                # while accounding for resuming a run.
                resume_outputs[run_output['ndead']] = run_output
                step_ndead.append(run_output['ndead'] - settings_dict['nlive'])
                if len(step_ndead) >= 2 and step_ndead[-1] == step_ndead[-2]:
                    add_points = False
                # store resume file in new file path
                shutil.copyfile(
                    root_name + '.resume',
                    root_name + '_' + str(step_ndead[-1]) + '.resume')
            except:  # pragma: no cover
                if comm is None or comm.Get_size() == 1:
                    raise
                else:
                    # print error info
                    traceback.print_exc(file=sys.stdout)
                    print('Error in process with rank == 0: '
                          'forcing MPI abort.')
                    sys.stdout.flush()  # Make sure message prints before abort
                    comm.Abort(1)
        if comm is not None:
            add_points = comm.bcast(add_points, root=0)
    if rank == 0:
        final_seed = settings_dict['seed']
    return step_ndead, resume_outputs, final_seed
