#!/usr/bin/env python
"""
Contains main function for running dynamic nested sampling.
"""
import copy
import os
import shutil
import warnings
import numpy as np
import scipy.signal
import nestcheck.data_processing
import nestcheck.io_utils
import nestcheck.write_polychord_output
import dyPolyChord.nlive_allocation
import dyPolyChord.output_processing


__all__ = ['run_dypolychord', 'check_settings']


@nestcheck.io_utils.timing_decorator
def run_dypolychord(run_polychord, dynamic_goal, settings_dict_in, **kwargs):
    """
    Performs dynamic nested sampling using the algorithm described in
    Appendix E of "Dynamic nested sampling: an improved algorithm for
    parameter estimation and evidence calculation" (Higson et al., 2018).
    This proceeds in 3 steps:

    1) Generate an initial run with a constant number of live points
    :math:`n_\\mathrm{init}`. This process is run in chuncks using PolyChord's
    max_ndead setting to allow periodic saving of .resume files so the initial
    run can be resumed at different points.
    2) Calculate an allocation of the number of live points at each likelihood
    for use in step 3. Also cleans up resume files and saves relevant
    information.
    3) Generate dynamic nested sampling run using the calculated live point
    allocation.
    4) Combine the initial and dynamic runs and write output files in the
    PolyChord format, and remove the intermediate output files produced.

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
    logl_warn_only: bool, optional
        Whether to raise error or warning if multiple samples have the same
        loglikelihood. This is passed to nestcheck's check_ns_run function;
        see its documentation for more details.
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
    logl_warn_only = kwargs.pop('logl_warn_only', False)
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
        # Make a copy of settings dic so we dont edit settings
        settings_dict = copy.deepcopy(settings_dict_in)
        settings_dict['file_root'] = settings_dict['file_root'] + '_init'
        settings_dict['nlive'] = ninit
    if dynamic_goal == 0:
        # We definitely won't need to resume midway through in this case, so
        # just run PolyChod normally
        run_polychord(settings_dict, comm=comm)
        if settings_dict['seed'] >= 0:
            final_seed = settings_dict['seed'] + seed_increment
        else:
            final_seed = settings_dict['seed']
    else:
        step_ndead, resume_outputs, final_seed = run_and_save_resumes(
            run_polychord, settings_dict, init_step, seed_increment, comm=comm)
    # Step 2: calculate an allocation of live points
    # ----------------------------------------------
    if rank == 0:
        init_run = nestcheck.data_processing.process_polychord_run(
            settings_dict_in['file_root'] + '_init',
            settings_dict_in['base_dir'])
        # Calculate max number of samples
        if settings_dict_in['max_ndead'] > 0:
            samp_tot = settings_dict_in['max_ndead']
            assert settings_dict_in['max_ndead'] > init_run['logl'].shape[0], (
                'all points used in init run - None left for dynamic run')
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
            dyn_info['resume_nlike'] = resume_outputs[resume_ndead]['nlike']
        nestcheck.io_utils.pickle_save(
            dyn_info, root_name + '_dyn_info', overwrite_existing=True)
        if dynamic_goal != 0:
            # Remove all the temporary resume files. Use set to avoid
            # duplicates as these cause OSErrors.
            for snd in set(step_ndead):
                os.remove(root_name + '_init_' + str(snd) + '.resume')
        # Step 3: do dynamic run
        # ----------------------
        # remove edits from init
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
        settings_dict['read_resume'] = bool(dyn_info['peak_start_ind'] != 0)
        settings_dict['file_root'] = settings_dict_in['file_root'] + '_dyn'
    run_polychord(settings_dict, comm=comm)
    if rank == 0:
        # Step 4: process output and tidy
        # -------------------------------
        # Combine initial and dynamic runs
        run = dyPolyChord.output_processing.process_dypolychord_run(
            settings_dict_in['file_root'], settings_dict_in['base_dir'],
            dynamic_goal=dynamic_goal, logl_warn_only=logl_warn_only)
        # Save combined output in PolyChord format
        nestcheck.write_polychord_output.write_run_output(
            run, logl_warn_only=logl_warn_only,
            stats_means_errs=stats_means_errs, **output_settings)
        if clean:
            # Remove temporary files
            clean_extra_output(root_name)


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
    seed: int
        Random seed. This is incremented after each run so it can be used
        when resuming without generating correlated points.
    """
    # set up rank if running with MPI
    settings_dict = copy.deepcopy(settings_dict_in)
    if comm is not None:
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
        if comm is not None:
            add_points = comm.bcast(add_points, root=0)
    return step_ndead, resume_outputs, settings_dict['seed']
