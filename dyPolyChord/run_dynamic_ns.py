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


def check_settings_dict(settings_dict_in):
    """
    Checks the input dictionary of PolyChord settings. Issues warnings where
    these are not appropriate, and adds default values.
    """
    default_settings = {'nlive': 100,
                        'num_repeats': 20,
                        'file_root': 'temp',
                        'base_dir': 'chains',
                        'seed': -1,
                        'do_clustering': True,
                        'max_ndead': -1}
    mandatory_settings = {'cluster_posteriors': False,
                          'nlives': {},
                          'write_dead': True,
                          'write_stats': True,
                          'write_paramnames': False,
                          'write_live': False,
                          'write_resume': False,
                          'read_resume': False,
                          'cluster_posteriors': False}
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
        if key in settings_dict_in:
            output_settings[key] = settings_dict_in[key]
        else:
            output_settings[key] = False
        settings_dict_in[key] = False
    return settings_dict, output_settings


@nestcheck.io_utils.timing_decorator
def run_dypolychord(run_func, dynamic_goal, settings_dict_in, **kwargs):
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
    PolyChord format.

    Parameters
    ----------
    run_func: function
        Function which runs PolyChord and takes a settings dictionary as its
        argument.
    dynamic_goal: float or int
        Number in (0, 1) which determines how to allocate computational effort
        between parameter estimation and evidence calculation. See the dynamic
        nested sampling paper for more details.
    settings_dict: dict
        PolyChord settings to use (see below for default settings).
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
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Step 1: do initial run
    # ----------------------
    # set up rank if running with MPI
    if comm is not None:
        rank = comm.Get_rank()
    else:
        rank = 0
    settings_dict = None  # define for rank != 1
    if rank == 0:
        settings_dict_in, output_settings = check_settings_dict(settings_dict_in)
        settings_dict = copy.deepcopy(settings_dict_in) # so we dont edit settings
        root = os.path.join(settings_dict['base_dir'],
                            settings_dict['file_root'])
        settings_dict['file_root'] = settings_dict['file_root'] + '_init'
        settings_dict['nlive'] = ninit
    if dynamic_goal == 0:
        run_func(settings_dict, comm=comm)
    else:
        if rank == 0:
            try:
                os.remove(root + '_init.resume')
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
                if settings_dict['seed'] >= 0:
                    settings_dict['seed'] += seed_increment
            run_func(settings_dict, comm=comm)
            if rank == 0:
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
                    root + '_init.resume',
                    root + '_init_' + str(step_ndead[-1]) + '.resume')
            if comm is not None:
                add_points = comm.bcast(add_points, root=0)
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
        if dyn_info['peak_start_ind'] != 0:
            # subtract 1 as ndead=1 corresponds to point 0
            resume_steps = np.asarray(step_ndead) - 1
            # Work out which resume file to load
            resume_ndead = step_ndead[np.where(
                resume_steps < dyn_info['peak_start_ind'])[0][-1]]
            # copy resume step to dynamic file root
            shutil.copyfile(root + '_init_' + str(resume_ndead) + '.resume',
                            root + '_dyn.resume')
            # Save resume info
            dyn_info['resume_ndead'] = resume_ndead
            dyn_info['resume_nlike'] = resume_outputs[resume_ndead]['nlike']
        nestcheck.io_utils.pickle_save(
            dyn_info, root + '_dyn_info', overwrite_existing=True)
        try:
            # Remove all the temporary resume files. Use set to avoid
            # duplicates as these cause OSErrors.
            for snd in set(step_ndead):
                os.remove(root + '_init_' + str(snd) + '.resume')
        except NameError:  # occurs when not saving resumes so step_ndead list
            pass
        # Step 3: do dynamic run
        # ----------------------
        final_init_seed = settings_dict['seed']
        # remove edits from init
        settings_dict = copy.deepcopy(settings_dict_in)
        if final_init_seed >= 0:
            assert settings_dict_in['seed'] >= 0, (
                'if input seed was <0 it should not have been edited')
            settings_dict['seed'] = final_init_seed + seed_increment
        if dyn_info['peak_start_ind'] != 0:
            settings_dict['nlive'] = ninit
        else:
            settings_dict['nlive'] = dyn_info['nlives_dict'][
                min(dyn_info['nlives_dict'].keys())]
        settings_dict['nlives'] = dyn_info['nlives_dict']
        settings_dict['read_resume'] = (dyn_info['peak_start_ind'] != 0)
        settings_dict['file_root'] = settings_dict_in['file_root'] + '_dyn'
    run_func(settings_dict, comm=comm)
    if rank == 0:
        # Step 4: process output and tidy
        # -------------------------------
        # Combine initial and dynamic runs
        run = dyPolyChord.output_processing.process_dypolychord_run(
            settings_dict_in['file_root'], settings_dict_in['base_dir'],
            dynamic_goal=dynamic_goal, logl_warn_only=logl_warn_only)
        for key in ['nlives_dict', 'dynamic_goal', 'dyn_nlike', 'resume_ndead',
                    'peak_start_ind', 'init_nlive_allocation_unsmoothed',
                    'init_nlike', 'init_nlive_allocation', 'resume_nlike']:
            run['output'].pop(key, None)
        # Save combined output in PolyChord format
        nestcheck.write_polychord_output.write_run_output(
            run, logl_warn_only=logl_warn_only,
            stats_means_errs=False,
            **output_settings)
        # Remove temporary files
        for extra in ['init', 'dyn']:
            os.remove(root + '_{0}.stats'.format(extra))
            os.remove(root + '_{0}_dead-birth.txt'.format(extra))
            # tidy up remaining .resume files (if the function has reach this
            # point, both the initial and dynamic runs have finished so we
            # shouldn't need to resume)
            try:
                os.remove(root + '_{0}.resume'.format(extra))
            except OSError:
                pass
