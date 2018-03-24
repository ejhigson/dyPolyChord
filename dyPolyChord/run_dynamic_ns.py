#!/usr/bin/env python
"""
Functions for using PyPolyChord, including using it to perform dynamic nested
sampling.
"""
import os
import time
import shutil
import copy
import numpy as np
import PyPolyChord
import nestcheck.io_utils as iou
import nestcheck.analyse_run as ar
import nestcheck.data_processing
# from mpi4py import MPI


def run_dypolychord_evidence(pc_settings_in, likelihood, prior, ndims,
                             **kwargs):
    """
    Dynamic nested sampling targeting increased evidence accuracy using
    polychord.

    dyn_nlive_step gives the fraction of points from the initial run to include
    in nlives. The dynamic run checks nlives to see if point should be added or
    removed every single step.

    The dynamic run's live points are determined by the dynamic ns importance
    calculation.
    Both the initial and dynamic runs use settings.nlive = ninit to determine
    clustering and resume writing.
    """
    ninit = kwargs.pop('ninit', 10)
    dyn_nlive_step = kwargs.pop('dyn_nlive_step', 1)
    nlive_const = kwargs.pop('nlive_const', pc_settings_in.nlive)
    nderived = kwargs.pop('nderived', 0)
    print_time = kwargs.pop('print_time', False)
    if 'init_step' in kwargs:
        # init_step not needed for dynamic ns targeting evidence
        kwargs.pop('init_step')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    start_time = time.time()
    assert not pc_settings_in.nlives
    assert not pc_settings_in.read_resume
    # Step 1: do initial run
    # ----------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # so we dont edit settings
    pc_settings.file_root = pc_settings_in.file_root + '_init'
    pc_settings.nlive = ninit
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    init_run = nestcheck.data_processing.process_polychord_run(
        pc_settings.file_root, pc_settings.base_dir)
    # Step 2: calculate an allocation of live points
    # ----------------------------------------------
    logx_init = ar.get_logx(init_run['nlive_array'])
    w_rel = ar.rel_posterior_mass(logx_init, init_run['logl'])
    w_rel = np.cumsum(w_rel)
    w_rel = w_rel.max() - w_rel
    w_rel /= np.abs(np.trapz(w_rel, x=logx_init))
    # calculate a distribution of nlive points in proportion to w_rel
    if pc_settings_in.max_ndead > 0:
        nlives_array = w_rel * (pc_settings_in.max_ndead -
                                init_run['logl'].shape[0])
    else:
        nlives_array = (w_rel * init_run['logl'].shape[0] *
                        (nlive_const - ninit) / ninit)
    assert nlives_array[0] == nlives_array.max()
    # convert to ints
    nlives_array = np.rint(nlives_array).astype(int)
    assert nlives_array.min() >= 0
    # get nlives dict
    nlives_dict = {}
    steps = dyn_nlive_step * np.asarray(range(w_rel.shape[0] //
                                              dyn_nlive_step))
    for step in steps:
        nlives_dict[init_run['logl'][step]] = int(nlives_array[step])
    # nlives_dict[-1e100] = int(nlives_array.max())
    # Step 3: do dynamic run
    # ----------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # remove edits from init
    pc_settings.nlives = nlives_dict
    pc_settings.file_root = pc_settings_in.file_root + '_dyn'
    # In order to start by sampling nlives_array.max() live points but do
    # clustering and resume writing with pc_settings.nlive = ninit we run the
    # first few steps then resume with pc_settings.nlive changed
    pc_settings.seed += 100
    pc_settings.nlive = nlives_array.max()
    pc_settings.max_ndead = pc_settings.nlive
    pc_settings.read_resume = False
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    # Now load with pc_settings.nlive = ninit. This is just for resume writing
    # and clustering - the number of live points is controlled by
    # pc_settings.nlives
    pc_settings.seed += 100
    pc_settings.nlive = ninit
    pc_settings.read_resume = True
    pc_settings.max_ndead = pc_settings_in.max_ndead
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
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


    dyn_nlive_step gives the fraction of points from the initial run to include
    in nlives. The dynamic run checks nlives to see if point should be added or
    removed every single step.

    The dynamic run's live points are determined by the dynamic ns importance
    calculation.
    Both the initial and dynamic runs use settings.nlive = ninit to determine
    clustering and resume writing.
    """
    ninit = kwargs.pop('ninit', 10)
    init_step = kwargs.pop('init_step', ninit)
    dyn_nlive_step = kwargs.pop('dyn_nlive_step', 1)
    nlive_const = kwargs.pop('nlive_const', pc_settings_in.nlive)
    nderived = kwargs.pop('nderived', 0)
    print_time = kwargs.pop('print_time', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
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
            if step_ndead[-1] == step_ndead[-2] + 1:
                add_points = False
    init_run = nestcheck.data_processing.process_polychord_run(
        pc_settings.file_root, pc_settings.base_dir)
    # Step 2: calculate an allocation of live points
    # ----------------------------------------------
    logx_init = ar.get_logx(init_run['nlive_array'])
    w_rel = ar.rel_posterior_mass(logx_init, init_run['logl'])
    # calculate a distribution of nlive points in proportion to w_rel
    if pc_settings_in.max_ndead > 0:
        nlives_array = w_rel * (pc_settings_in.max_ndead -
                                init_run['logl'].shape[0])
    else:
        nlives_array = (w_rel * init_run['logl'].shape[0] *
                        (nlive_const - ninit) / ninit)
    nlives_array = np.rint(nlives_array).astype(int)
    # make sure it does not dip below ninit until it reaches the peak
    peak_start_ind = np.where(nlives_array >= ninit)[0][0]
    nlives_array[:peak_start_ind] = ninit
    assert nlives_array.min() >= 0
    # get nlives dict
    nlives_dict = {-1e100: ninit}
    steps = dyn_nlive_step * np.asarray(range(w_rel.shape[0] //
                                              dyn_nlive_step))
    for i, step in enumerate(steps[:-1]):
        # Assign nlives to the logl one step before to make sure the number of
        # live points is increased in time
        nlives_dict[init_run['logl'][step]] = int(nlives_array[steps[i + 1]])
    # subtract 1 as ndead=1 corresponds to point 0
    resume_steps = np.asarray(step_ndead) - 1
    # Load the last resume before we reach the peak
    resume_ndead = step_ndead[np.where(
        resume_steps < peak_start_ind)[0][-1]]
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
    pc_settings.nlives = nlives_dict
    pc_settings.read_resume = True
    pc_settings.file_root = pc_settings_in.file_root + '_dyn'
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
    # Save info about where the dynamic run was resumed from
    dyn_info = {'resume_ndead': resume_ndead,
                'resume_nlike': run_outputs_at_resumes[resume_ndead].nlike}
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