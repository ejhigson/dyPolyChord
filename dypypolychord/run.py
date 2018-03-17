#!/usr/bin/envython
"""
Functions for using PyPolyChord, including using it to perform dynamic nested
sampling.
"""
import os
import time
import shutil
import copy
import numpy as np
# from mpi4py import MPI
import PyPolyChord
import nestcheck.analyse_run as ar
import nestcheck.data_processing
import dypypolychord.save_load_utils


def run_standard_polychord(pc_settings, likelihood, prior, ndims, **kwargs):
    """
    Wrapper function of same format as run_dynamic_polychord for running
    standard polychord runs.
    """
    # comm = MPI.COMM_WORLD
    nderived = kwargs.pop('nderived', 0)
    print_time = kwargs.pop('print_time', False)
    assert not pc_settings.nlives, (
        'nlive should not change in standard nested sampling! nlives=' +
        str(pc_settings.nlives))
    start_time = time.time()
    # do initial run
    # --------------
    output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                       pc_settings, prior)
    # if comm.rank == 0:
    dypypolychord.save_load_utils.save_info(pc_settings, output)
    if print_time:
        end_time = time.time()
        print('#####################################')
        print('run_standard_polychord took %.3f sec' % (end_time - start_time))
        print('file_root = ' + pc_settings.file_root)
        print('#####################################')


def run_dynamic_polychord_evidence(pc_settings_in, likelihood, prior, ndims,
                                   **kwargs):
    """
    Dynamic nested sampling using polychord.
    """
    # comm = MPI.COMM_WORLD
    ninit = kwargs.pop('ninit', 10)
    dyn_nlive_step = kwargs.pop('dyn_nlive_step', 10)
    nlive_const = kwargs.pop('nlive_const', pc_settings_in.nlive)
    # n_samples_max = kwargs.pop('n_samples_max', None)
    nderived = kwargs.pop('nderived', 0)
    kwargs.pop('init_step', None)
    print_time = kwargs.pop('print_time', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    # if comm.rank == 0:
    start_time = time.time()
    assert not pc_settings_in.nlives
    assert not pc_settings_in.read_resume
    # do initial run
    # --------------
    pc_settings = copy.deepcopy(pc_settings_in)  # so we dont edit settings
    pc_settings.file_root = pc_settings_in.file_root + '_init'
    pc_settings.nlive = ninit
    init_output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                            pc_settings, prior)
    dypypolychord.save_load_utils.save_info(
        pc_settings, init_output)
    init_run = nestcheck.data_processing.process_polychord_run(
        pc_settings.base_dir + '/' + pc_settings.file_root)
    # Work out a new allocation of live points
    # ----------------------------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # remove edits from init
    pc_settings.seed += 100
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
    # update settings for the dynamic step
    pc_settings.nlives = nlives_dict
    pc_settings.nlive = nlives_array.max()
    pc_settings.resume = False
    pc_settings.file_root = pc_settings_in.file_root + '_dyn'
    # # broadcast dynamic settings to other threads
    dyn_output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                           pc_settings, prior)
    # if comm.rank == 0:
    dypypolychord.save_load_utils.save_info(
        pc_settings, dyn_output)
    if print_time:
        end_time = time.time()
        print('#############################################')
        print('run_dynamic_polychord_evidence took %.3f sec' %
              (end_time - start_time))
        print('file_root = ' + pc_settings_in.file_root)
        print('#############################################')


def run_dynamic_polychord_param(pc_settings_in, likelihood, prior, ndims,
                                **kwargs):
    """
    Dynamic nested sampling using polychord.
    """
    # comm = MPI.COMM_WORLD
    ninit = kwargs.pop('ninit', 10)
    init_step = kwargs.pop('init_step', ninit)
    dyn_nlive_step = kwargs.pop('dyn_nlive_step', 1)
    nlive_const = kwargs.pop('nlive_const', pc_settings_in.nlive)
    # n_samples_max = kwargs.pop('n_samples_max', None)
    nderived = kwargs.pop('nderived', 0)
    print_time = kwargs.pop('print_time', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    start_time = time.time()
    assert not pc_settings_in.nlives
    assert not pc_settings_in.read_resume
    # do initial run
    # --------------
    pc_settings = copy.deepcopy(pc_settings_in)  # so we dont edit settings
    pc_settings.file_root = pc_settings_in.file_root + '_init'
    pc_settings.nlive = ninit
    pc_settings.write_resume = True
    pc_settings.read_resume = False
    add_points = True
    step_ndead = []
    # runs_at_resumes = {}
    run_outputs_at_resumes = {}
    while add_points:
        if len(step_ndead) == 1:
            pc_settings.read_resume = True
        pc_settings.max_ndead = (len(step_ndead) + 1) * init_step
        pc_settings.seed += 100
        output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                           pc_settings, prior)
        # store run for use getting nlike
        run_outputs_at_resumes[output.ndead] = output
        step_ndead.append(output.ndead - pc_settings.nlive)
        # # Check the run at the resume point is as expected
        # run = nestcheck.data_processing.process_polychord_run(
        #     pc_settings.base_dir + '/' + pc_settings.file_root)
        # assert run['thread_labels'].shape[0] == output.ndead
        # assert np.unique(run['thread_labels']).shape[0] == ninit
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
        pc_settings.base_dir + '/' + pc_settings.file_root)
    # Work out a new allocation of live points
    # ----------------------------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # remove edits from init
    pc_settings.seed += 100
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
    pc_settings.nlive = dyn_nlive_step
    pc_settings.nlives = nlives_dict
    pc_settings.read_resume = True
    # copy resume step to dynamic file root
    shutil.copyfile(pc_settings.base_dir + '/' + pc_settings.file_root +
                    '_init_' + str(resume_ndead) + '.resume',
                    pc_settings.base_dir + '/' +
                    pc_settings.file_root + '_dyn.resume')
    # Remove the mess of other resume files
    # update settings for the dynamic step
    pc_settings.file_root = pc_settings_in.file_root + '_dyn'
    dyn_output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                           pc_settings, prior)
    for snd in step_ndead:
        os.remove(pc_settings_in.base_dir + '/' +
                  pc_settings_in.file_root +
                  '_init_' + str(snd) + '.resume')
    dypypolychord.save_load_utils.save_info(
        pc_settings, dyn_output, resume_ndead=resume_ndead,
        resume_nlike=run_outputs_at_resumes[resume_ndead].nlike)

    if print_time:
        end_time = time.time()
        print('##########################################')
        print('run_dynamic_polychord_param took %.3f sec' %
              (end_time - start_time))
        print('file_root = ' + pc_settings_in.file_root)
        print('##########################################')