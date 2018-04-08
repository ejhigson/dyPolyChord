#!/usr/bin/env python
"""
Functions for using PyPolyChord, including using it to perform dynamic nested
sampling.
"""
import os
import time
import shutil
import copy
import itertools
import numpy as np
import scipy.signal
import PyPolyChord
import nestcheck.io_utils as iou
import nestcheck.analyse_run as ar
import nestcheck.data_processing


def run_dypolychord_evidence(pc_settings_in, likelihood, prior, ndims,
                             **kwargs):
    """
    Dynamic nested sampling targeting increased evidence accuracy using
    polychord.

    PolyChord checks nlives to see if point should be added or
    removed every single step.

    The dynamic run's live points are determined by the dynamic ns importance
    calculation.
    Both the initial and dynamic runs use settings.nlive = ninit to determine
    clustering and resume writing.
    """
    ninit = kwargs.pop('ninit', 10)
    nlive_const = kwargs.pop('nlive_const', pc_settings_in.nlive)
    nderived = kwargs.pop('nderived', 0)
    print_time = kwargs.pop('print_time', False)
    if 'init_step' in kwargs:
        # init_step not needed for dynamic ns targeting evidence
        kwargs.pop('init_step')
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    start_time = time.time()
    assert not pc_settings_in.nlives
    assert not pc_settings_in.read_resume
    # Step 1: do initial run
    # ----------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # so we dont edit settings
    pc_settings.file_root = pc_settings_in.file_root + '_init'
    pc_settings.nlive = ninit
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    # Step 2: calculate an allocation of live points
    # ----------------------------------------------
    dyn_info = nlive_allocation(pc_settings_in, ninit, nlive_const, 0)
    # Step 3: do dynamic run
    # ----------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # remove edits from init
    pc_settings.nlives = dyn_info['nlives_dict']
    pc_settings.file_root = pc_settings_in.file_root + '_dyn'
    # In order to start by sampling nlives_array.max() live points but do
    # clustering and resume writing with pc_settings.nlive = ninit we run the
    # first few steps then resume with pc_settings.nlive changed
    pc_settings.seed += 100
    pc_settings.nlive = max(dyn_info['nlives_dict'].values())
    # ######################################################
    # Commented out save and load due for now to memory leak
    # replaced with below line
    # ######################################################
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    # ######################################################
    # pc_settings.max_ndead = pc_settings.nlive
    # pc_settings.write_resume = True
    # pc_settings.read_resume = False
    # PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    # # Now load with pc_settings.nlive = ninit. This is just for resume writing
    # # and clustering - the number of live points is controlled by
    # # pc_settings.nlives
    # pc_settings.seed += 100
    # pc_settings.nlive = min(ninit, max(nlives_dict.values()))
    # pc_settings.read_resume = True
    # pc_settings.max_ndead = pc_settings_in.max_ndead
    # PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    # ######################################################
    # Save info about the dynamic run
    iou.pickle_save(dyn_info,
                    (pc_settings_in.base_dir + '/' +
                     pc_settings_in.file_root + '_dyn_info'),
                    overwrite_existing=True)
    if print_time:
        end_time = time.time()
        print('#############################################')
        print('run_dypolychord_evidence took %.3f sec' %
              (end_time - start_time))
        print('file_root = ' + pc_settings_in.file_root)
        print('#############################################')


def run_dypolychord_param(pc_settings_in, likelihood, prior, ndims, **kwargs):
    """
    Dynamic nested sampling targeting increased parameter estimation accuracy
    using polychord.

    PolyChord checks nlives to see if point should be added or
    removed every single step.

    The dynamic run's live points are determined by the dynamic ns importance
    calculation.
    Both the initial and dynamic runs use settings.nlive = ninit to determine
    clustering and resume writing.
    """
    ninit = kwargs.pop('ninit', 10)
    init_step = kwargs.pop('init_step', ninit)
    nlive_const = kwargs.pop('nlive_const', pc_settings_in.nlive)
    nderived = kwargs.pop('nderived', 0)
    print_time = kwargs.pop('print_time', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    start_time = time.time()
    assert not pc_settings_in.nlives
    assert not pc_settings_in.read_resume
    # Step 1: do initial run
    # ----------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # so we dont edit settings
    pc_settings.file_root = pc_settings_in.file_root + '_init'
    pc_settings.nlive = ninit
    pc_settings.write_resume = True
    pc_settings.read_resume = False
    add_points = True
    step_ndead = []
    run_outputs_at_resumes = {}
    while add_points:
        if len(step_ndead) == 1:
            pc_settings.read_resume = True
        pc_settings.max_ndead = (len(step_ndead) + 1) * init_step
        pc_settings.seed += 100
        output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                           pc_settings, prior)
        # store run outputs for use getting nlike
        run_outputs_at_resumes[output.ndead] = output
        step_ndead.append(output.ndead - pc_settings.nlive)
        # store resume file in new file path
        shutil.copyfile(pc_settings.base_dir + '/' +
                        pc_settings.file_root + '.resume',
                        pc_settings.base_dir + '/' +
                        pc_settings.file_root +
                        '_' + str(step_ndead[-1]) + '.resume')
        if len(step_ndead) >= 2:
            if step_ndead[-1] <= step_ndead[-2] + 1:
                add_points = False
    # Step 2: calculate an allocation of live points
    # ----------------------------------------------
    # calculate a distribution of nlive points in proportion to w_rel
    dyn_info = nlive_allocation(pc_settings_in, ninit, nlive_const, 1)
    # subtract 1 as ndead=1 corresponds to point 0
    resume_steps = np.asarray(step_ndead) - 1
    # Load the last resume before we reach the peak
    resume_ndead = step_ndead[np.where(
        resume_steps < dyn_info['peak_start_ind'])[0][-1]]
    # copy resume step to dynamic file root
    shutil.copyfile(pc_settings_in.base_dir + '/' + pc_settings_in.file_root +
                    '_init_' + str(resume_ndead) + '.resume',
                    pc_settings_in.base_dir + '/' +
                    pc_settings_in.file_root + '_dyn.resume')
    # Remove all the temporary resume files
    for snd in step_ndead:
        os.remove(pc_settings_in.base_dir + '/' +
                  pc_settings_in.file_root +
                  '_init_' + str(snd) + '.resume')
    # Step 3: do dynamic run
    # ----------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # remove edits from init
    pc_settings.seed += 100
    pc_settings.nlive = ninit
    pc_settings.nlives = dyn_info['nlives_dict']
    pc_settings.read_resume = True
    pc_settings.file_root = pc_settings_in.file_root + '_dyn'
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    # Save info about where the dynamic run was resumed from
    dyn_info['resume_ndead'] = resume_ndead
    dyn_info['resume_nlike'] = run_outputs_at_resumes[resume_ndead].nlike
    iou.pickle_save(dyn_info,
                    (pc_settings_in.base_dir + '/' +
                     pc_settings_in.file_root + '_dyn_info'),
                    overwrite_existing=True)
    if print_time:
        end_time = time.time()
        print('##########################################')
        print('run_dypolychord_param took {} sec'
              .format(end_time - start_time))
        print('file_root = ' + pc_settings_in.file_root)
        print('##########################################')


# Helper functions
# ----------------


def nlive_allocation(pc_settings_in, ninit, nlive_const, dynamic_goal,
                     **kwargs):
    """
    Loads initial run and calculates an allocation of life points for dynamic
    run.
    """
    assert dynamic_goal in [0, 1]
    default_smoothing = (lambda x: scipy.signal.savgol_filter(
        x, 1 + (2 * ninit), 3, mode='nearest'))
    smoothing_filter = kwargs.pop('smoothing_filter', default_smoothing)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    init_run = nestcheck.data_processing.process_polychord_run(
        pc_settings_in.file_root + '_init', pc_settings_in.base_dir)
    logx_init = ar.get_logx(init_run['nlive_array'])
    w_rel = ar.rel_posterior_mass(logx_init, init_run['logl'])
    # Calculate the importance of each point
    if dynamic_goal == 0:
        imp = np.cumsum(w_rel)
        imp = imp.max() - imp
        assert imp[0] == imp.max()
    elif dynamic_goal == 1:
        imp = w_rel
    imp /= np.abs(np.trapz(imp, x=logx_init))
    assert imp.min() >= 0
    # Calculate number of samples and multiply it by imp to get nlive
    if pc_settings_in.max_ndead > 0:
        samp_remain = pc_settings_in.max_ndead - init_run['logl'].shape[0]
        assert samp_remain > 0, (
            'all points used in initial run and none left for dynamic run!')
    else:
        samp_remain = init_run['logl'].shape[0] * ((nlive_const / ninit) - 1)
    nlives_unsmoothed = imp * samp_remain
    # Smooth and round nlive
    if smoothing_filter is not None:
        # Perform smoothing *before* rounding to nearest integer
        nlives = np.rint(smoothing_filter(nlives_unsmoothed))
    else:
        nlives = np.rint(nlives_unsmoothed)
    nlives_unsmoothed = np.rint(nlives_unsmoothed)
    if dynamic_goal == 1:
        # make sure nlives does not dip below ninit until it reaches the peak
        peak_start_ind = np.where(nlives >= ninit)[0][0]
        nlives[:peak_start_ind] = ninit
        # for diagnostics on smooothing, lets give the unsmoothed nlives the
        # same treatment
        nlives_unsmoothed[:np.where(nlives_unsmoothed >= ninit)[0][0]] = ninit
    # Get the indexes of nlives points which are different to the previous
    # points (i.e. remove consecutive duplicates, keeping first occurance)
    inds_to_use = np.concatenate((np.asarray([0]),
                                  np.where(np.diff(nlives) != 0)[0] + 1))
    # ################################################
    # Perform some checks
    assert np.all(np.diff(init_run['logl']) > 0)
    assert (count_turning_points(nlives) ==
            count_turning_points(nlives[inds_to_use]))
    if dynamic_goal == 0:
        assert nlives[0] == nlives.max()
        assert np.all(np.diff(nlives) <= 0), (
            'When targeting evidence, nlive should monotincally decrease!'
            + ' nlives = ' + str(nlives))
        assert np.all(np.diff(nlives[inds_to_use]) < 0), (
            'When targeting evidence, nlive should monotincally decrease!'
            + ' nlives = ' + str(nlives))
    elif dynamic_goal == 1:
        assert nlives[0] == ninit
        msg = str(count_turning_points(nlives)) + ' turning points in nlive.'
        if smoothing_filter is None:
            msg += ' Smoothing_filter is None.'
        else:
            msg += ' Without smoothing_filter there would have been '
            msg += str(count_turning_points(nlives_unsmoothed))
        print(msg)
    # ################################################
    # Check logl = approx -inf is mapped to the starting number of live points
    nlives_dict = {-1.e100: int(nlives[0])}
    for ind in inds_to_use:
        nlives_dict[init_run['logl'][ind]] = int(nlives[ind])
    # Store the nlive allocations for dyn_info
    dyn_info = {'init_nlive_allocation': nlives,
                'init_nlive_allocation_unsmoothed': nlives_unsmoothed,
                'nlives_dict': nlives_dict}
    if dynamic_goal == 1:
        dyn_info['peak_start_ind'] = peak_start_ind
    return dyn_info


def count_turning_points(array):
    """Returns number of turning points in a 1d array."""
    # remove adjacent duplicates only
    uniq = np.asarray([k for k, g in itertools.groupby(array)])
    return (np.diff(np.sign(np.diff(uniq))) != 0).sum()
