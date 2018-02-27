#!/usr/bin/env python
"""
Functions for using PyPolyChord, including using it to perform dynamic nested
sampling.
"""
import os
import functools
import time
import shutil
import copy
import numpy as np
import PyPolyChord
from PyPolyChord.priors import UniformPrior
import PyPolyChord.likelihoods
from mpi4py import MPI
import nestcheck.analyse_run as ar
import nestcheck.io_utils as iou
import nestcheck.data_processing as dp


def save_info(settings, output, resume_ndead=None):
    """
    Save settings and output information using pickle and a standard file
    name based on the settings' file root.
    """
    info_to_save = {'output': output.__dict__, 'settings': settings.__dict__}
    if resume_ndead is not None:
        assert settings.nlives
        info_to_save['settings']['resume_ndead'] = resume_ndead
    iou.pickle_save(info_to_save,
                    'chains/' + settings.file_root + '_info',
                    print_time=False, print_filename=False,
                    overwrite_existing=True)


def settings_root(pc_settings):
    """Get a standard string containing information about settings."""
    root = str(pc_settings.grade_dims[0]) + 'd'
    root += '_' + str(pc_settings.nlive) + 'nlive'
    root += '_' + str(pc_settings.num_repeats) + 'nrepeats'
    return root


def uniform_prior(hypercube, ndims=2, prior_thetamax=5):
    """ Uniform prior. """

    theta = [0.0] * ndims
    for i, x in enumerate(hypercube):
        theta[i] = UniformPrior(-prior_thetamax, prior_thetamax)(x)

    return theta


def get_w_rel(run):
    """Get relative weight of points in a nested sampling run."""
    logx = ar.get_logx(run['nlive_array'])
    logw = logx + run['logl']
    w_rel = np.exp(logw - logw.max())
    w_rel /= np.trapz(w_rel[::-1], x=logx[::-1])
    return w_rel


def run_standard_polychord(pc_settings, **kwargs):
    """
    Wrapper function of same format as run_dynamic_polychord for running
    standard polychord runs.
    """
    comm = MPI.COMM_WORLD
    ndims = kwargs.pop('ndims', 2)
    nderived = kwargs.pop('nderived', 0)
    likelihood = kwargs.pop('likelihood', PyPolyChord.likelihoods.gaussian)
    prior = kwargs.pop('prior', functools.partial(uniform_prior,
                                                  ndims=ndims,
                                                  prior_thetamax=5))
    assert not pc_settings.nlives
    start_time = time.time()
    # do initial run
    # --------------
    output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                       pc_settings, prior)
    if comm.rank == 0:
        save_info(pc_settings, output)
        end_time = time.time()
        print('####################################')
        print('run_standard_polychord took %.3f sec' % (end_time - start_time))
        print('####################################')


def run_dynamic_polychord(pc_settings_in, **kwargs):
    """
    Dynamic nested sampling using polychord.
    """
    comm = MPI.COMM_WORLD
    ninit = kwargs.pop('ninit', 10)
    init_step = kwargs.pop('init_step', ninit)
    dyn_nlive_step = kwargs.pop('dyn_nlive_step', 1)
    nlive_const = kwargs.pop('nlive_const', pc_settings_in.nlive)
    # n_samples_max = kwargs.pop('n_samples_max', None)
    ndims = kwargs.pop('ndims', 2)
    nderived = kwargs.pop('nderived', 0)
    likelihood = kwargs.pop('likelihood', PyPolyChord.likelihoods.gaussian)
    prior = kwargs.pop('prior', functools.partial(uniform_prior,
                                                  ndims=ndims,
                                                  prior_thetamax=5))
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
    pc_settings.write_resume = True
    pc_settings.read_resume = False
    add_points = True
    step_ndead = []
    runs_at_resumes = {}
    while add_points:
        if len(step_ndead) == 1:
            pc_settings.read_resume = True
        pc_settings.max_ndead = (len(step_ndead) + 1) * init_step
        output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                           pc_settings, prior)
        step_ndead.append(output.ndead - pc_settings.nlive)
        # TESTING
        run = dp.process_polychord_run(pc_settings.base_dir + '/' +
                                       pc_settings.file_root)
        assert run['thread_labels'].shape[0] == output.ndead
        assert np.unique(run['thread_labels']).shape[0] == ninit
        runs_at_resumes[output.ndead] = run
        # END TESTING
        shutil.copyfile(pc_settings.base_dir + '/' +
                        pc_settings.file_root + '.resume',
                        pc_settings.base_dir + '/' +
                        pc_settings.file_root +
                        '_' + str(step_ndead[-1]) + '.resume')
        if len(step_ndead) >= 2:
            if step_ndead[-1] == step_ndead[-2] + 1:
                add_points = False
    init_run = dp.process_polychord_run(pc_settings.base_dir + '/' +
                                        pc_settings.file_root)
    # Work out a new allocation of live points
    # ----------------------------------------
    pc_settings = copy.deepcopy(pc_settings_in)  # remove edits from init
    w_rel = get_w_rel(init_run)
    # calculate a distribution of nlive points in proportion to w_rel
    if pc_settings_in.max_ndead > 0:
        nlives_array = w_rel * (pc_settings_in.max_ndead -
                                init_run['logl'].shape[0])
    else:
        nlives_array = (w_rel * init_run['logl'].shape[0] *
                        (nlive_const - ninit) / ninit)
    # make sure it does not dip below ninit until it reaches the peak
    peak_start_ind = np.where(nlives_array > ninit)[0][0]
    nlives_array[:peak_start_ind] = ninit
    # get nlives dict
    nlives_dict = {-1. * 10e100: ninit}
    steps = dyn_nlive_step * np.asarray(range(w_rel.shape[0] //
                                              dyn_nlive_step))
    for i, step in enumerate(steps[:-1]):
        # Assign nlives to the logl one step before to make sure the number of
        # live points is increased in time
        nlive_i = int(nlives_array[steps[i + 1]])
        if nlive_i >= 1:
            nlives_dict[init_run['logl'][step]] = nlive_i
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
    # # broadcast dynamic settings to other threads
    # comm.Barrier()
    # pc_settings = comm.bcast(pc_settings, root=0)
    dyn_output = PyPolyChord.run_polychord(likelihood, ndims, nderived,
                                           pc_settings, prior)
    if comm.rank == 0:
        for snd in step_ndead:
            os.remove(pc_settings_in.base_dir + '/' +
                      pc_settings_in.file_root +
                      '_init_' + str(snd) + '.resume')
        save_info(pc_settings, dyn_output, resume_ndead=resume_ndead)
        end_time = time.time()
        print('####################################')
        print('run_dynamic_polychord took %.3f sec' % (end_time - start_time))
        print('####################################')


def get_polychord_data(file_root, n_runs, **kwargs):
    """
    Load and process polychord chains
    """
    data_dir = kwargs.pop('data_dir', 'cache/')
    chains_dir = kwargs.pop('chains_dir', 'chains/')
    load = kwargs.pop('load', False)
    save = kwargs.pop('save', False)
    dynamic = kwargs.pop('dynamic', False)
    overwrite_existing = kwargs.pop('overwrite_existing', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    save_name = file_root + '_' + str(n_runs) + 'runs'
    if load:
        try:
            return iou.pickle_load(data_dir + save_name)
        except OSError:  # FileNotFoundError is a subclass of OSError
            pass
    data = []
    errors = {}
    # load and process chains
    for i in range(1, n_runs + 1):
        try:
            root = chains_dir + file_root + '_' + str(i)
            if dynamic:
                data.append(process_dyn_run(root))
            else:
                data.append(dp.process_polychord_run(root))
        except (OSError, AssertionError, KeyError) as err:
            try:
                errors[type(err).__name__].append(i)
            except KeyError:
                errors[type(err).__name__] = [i]
    for error_name, val_list in errors.items():
        if val_list:
            save = False  # only save if every file is processed ok
            print(error_name + ' processing ' + str(len(val_list)) + ' / ' +
                  str(n_runs) + ' files')
            if len(val_list) != n_runs:
                print('Runs with errors were: ' + str(val_list))
    if save:
        print('Processed new chains: saving to ' + save_name)
        iou.pickle_save(data, data_dir + save_name, print_time=False,
                        overwrite_existing=overwrite_existing)
    return data


def process_dyn_run(root):
    """
    Process dynamic nested sampling run including both initial exploratory run
    and second dynamic run.
    """
    init = dp.process_polychord_run(root + '_init')
    dyn = dp.process_polychord_run(root + '_dyn')
    # Do some tests
    assert np.all(init['thread_min_max'][:, 0] == -np.inf)
    resume_ndead = dyn['settings']['resume_ndead']
    assert np.array_equal(init['logl'][:resume_ndead],
                          dyn['logl'][:resume_ndead])
    # Now lets remove the points which are in both dyn and init from init
    init['theta'] = init['theta'][resume_ndead:, :]
    for key in ['nlive_array', 'logl', 'thread_labels']:
        init[key] = init[key][resume_ndead:]
    # Check that at least one point in each thread (the point which was live at
    # the resume) remains
    assert np.array_equal(
        np.asarray(range(1, init['thread_min_max'].shape[0] + 1)),
        np.unique(init['thread_labels']))
    # We also need to remove the points that were live when the resume file was
    # written, as these show up as dead points in dyn
    live_inds = []
    empty_thread_inds = []
    for i, th_lab in enumerate(np.unique(init['thread_labels'])):
        th_inds = np.where(init['thread_labels'] == th_lab)[0]
        live_inds.append(th_inds[0])
        live_logl = init['logl'][th_inds[0]]
        if th_inds.shape[0] == 1:
            print('Empty thread created')
            empty_thread_inds.append(i)
        assert np.where(dyn['logl'] == live_logl)[0].shape == (1,), \
            'this point should be present in dyn too!'
        init['thread_min_max'][i, 0] = live_logl
    # lets remove the live points at init
    init['theta'] = np.delete(init['theta'], live_inds, axis=0)
    for key in ['nlive_array', 'logl', 'thread_labels']:
        init[key] = np.delete(init[key], live_inds)
    # remove any empty threads from logl_min_max
    init['thread_min_max'] = np.delete(
        init['thread_min_max'], empty_thread_inds, axis=0)
    # Combine the threads from dyn and init
    run = ar.combine_threads(ar.get_run_threads(dyn) +
                             ar.get_run_threads(init),
                             assert_birth_point=True)
    run['settings'] = {'resume_ndead': resume_ndead}
    return run
