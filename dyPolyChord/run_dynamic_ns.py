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
import dyPolyChord.nlive_allocation


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
    dyn_info = dyPolyChord.nlive_allocation.allocate(
        pc_settings_in, ninit, nlive_const, 0)
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
    PyPolyChord.run_polychord(likelihood, ndims, nderived, pc_settings, prior)
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
        if len(step_ndead) >= 2:
            if step_ndead[-1] == step_ndead[-2]:
                break
        # store resume file in new file path
        shutil.copyfile(pc_settings.base_dir + '/' +
                        pc_settings.file_root + '.resume',
                        pc_settings.base_dir + '/' +
                        pc_settings.file_root +
                        '_' + str(step_ndead[-1]) + '.resume')
    # Step 2: calculate an allocation of live points
    # ----------------------------------------------
    # calculate a distribution of nlive points in proportion to w_rel
    dyn_info = dyPolyChord.nlive_allocation.allocate(
        pc_settings_in, ninit, nlive_const, 1)
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
    # Remove all the temporary resume files. Use set to avoid duplicates as
    # these cause OSErrors.
    for snd in set(step_ndead):
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
