#!/usr/bin/env python
"""
Functions for loading and processing dyPolyChord dynamic nested sampling runs.
"""
import os
import warnings
import numpy as np
import nestcheck.ns_run_utils
import nestcheck.data_processing
import nestcheck.io_utils as iou


def settings_root(likelihood_name, prior_name, ndim, **kwargs):
    """
    Returns a standard string containing information about settings.

    Parameters
    ----------
    likelihood_name: str
    prior_name: str
    ndim: int
    prior_scale: float or int
    dynamic_goal: float, int or None
    nlive_const: int
    nrepeats: int
    nint: int, optional
    init_step: int, optional

    Returns
    -------
    root: str
    """
    prior_scale = kwargs.pop('prior_scale')
    dynamic_goal = kwargs.pop('dynamic_goal')
    nlive_const = kwargs.pop('nlive_const')
    nrepeats = kwargs.pop('nrepeats')
    ninit = kwargs.pop('ninit', None)
    init_step = kwargs.pop('init_step', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    root = '{}_{}_{}_dg{}'.format(likelihood_name, prior_name, prior_scale,
                                  dynamic_goal)
    if dynamic_goal is not None:
        assert ninit is not None
        root += '_{}init'.format(ninit)
        if dynamic_goal != 0:
            assert init_step is not None
            root += '_{}is'.format(init_step)
    root += '_{}d_{}nlive_{}nrepeats'.format(ndim, nlive_const, nrepeats)
    return root.replace('.', '_')


def process_dypolychord_run(file_root, base_dir, **kwargs):
    """
    Load the output files of a dynamic run and process them to the nestcheck
    format.

    Parameters
    ----------
    file_root: str
    base_dir: str
    dynamic_goal: float
    dup_assert: bool, optional
        Whether to throw an AssertionError if there are duplicate point
        loglikelihood values.
    dup_warn: bool, optional
        Whether to give a UserWarning if there are duplicate point
        loglikelihood values.

    Returns
    -------
    run: dict
        Nested sampling run in nestcheck format (see
        http://nestcheck.readthedocs.io/en/latest/api.html for more
        information).
    """
    dynamic_goal = kwargs.pop('dynamic_goal')
    dup_assert = kwargs.pop('dup_assert', False)
    dup_warn = kwargs.pop('dup_warn', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    init = nestcheck.data_processing.process_polychord_run(
        file_root + '_init', base_dir, dup_assert=dup_assert,
        dup_warn=dup_warn)
    assert np.all(init['thread_min_max'][:, 0] == -np.inf), (
        'Initial run contains threads not starting at -inf.\n'
        'thread_min_max=' + str(init['thread_min_max']))
    dyn = nestcheck.data_processing.process_polychord_run(
        file_root + '_dyn', base_dir, dup_assert=dup_assert,
        dup_warn=dup_warn)
    dyn_info = iou.pickle_load(os.path.join(
        base_dir, file_root + '_dyn_info'))
    if dynamic_goal == 0:
        # If dynamic_goal == 0 then nlive should only decrease, so check all
        # threads start by sampling
        assert np.all(dyn['thread_min_max'][:, 0] == -np.inf), (
            str(dyn['thread_min_max']))
    # Get info to run
    run_output = {'file_root': file_root,
                  'base_dir': base_dir}
    if 'resume_ndead' not in dyn_info:
        # The dynamic run was not resumed part way through the initial run:
        # hence there are no samples repeated in both runs' files and we can
        # simply combine dyn and init using standard nestcheck functions.
        run = nestcheck.ns_run_utils.combine_ns_runs([init, dyn])
        try:
            run_output['nlike'] = (
                init['output']['nlike'] + dyn['output']['nlike'])
        except KeyError:
            pass # protect from error reading nlike from .stats file
    else:
        # The dynamic run started by resuming the initial run part way
        # through, and we need to remove duplicate points from the combined run
        run = combine_resumed_dyn_run(init, dyn, dyn_info['resume_ndead'])
        try:
            run_output['nlike'] = (
                init['output']['nlike'] + dyn['output']['nlike']
                - dyn_info['resume_nlike'])
        except KeyError:
            pass # protect from error reading nlike from .stats file
    run['output'] = run_output
    # check the nested sampling run has the expected properties
    nestcheck.ns_run_utils.check_ns_run(
        run, dup_assert=dup_assert, dup_warn=dup_warn)
    return run


def combine_resumed_dyn_run(init, dyn, resume_ndead):
    """
    Merge initial run and dynamic run which was resumed from it, including
    removing duplicate points present in both runs.

    Parameters
    ----------
    init: dict
        Initial exploratory run in nestcheck format (see
        http://nestcheck.readthedocs.io/en/latest/api.html for more
        information).
    dyn: dict
        Dynamic run in nestcheck format.
    resume_ndead: int
        The number of dead points present when dyn was resumed from init.

    Returns
    -------
    run: dict
        Combined run in nestcheck format.
    """
    assert np.array_equal(
        init['logl'][:resume_ndead], dyn['logl'][:resume_ndead]), (
            'The first {0} points should be the same'.format(resume_ndead))
    init['theta'] = init['theta'][resume_ndead:, :]
    for key in ['nlive_array', 'logl', 'thread_labels']:
        init[key] = init[key][resume_ndead:]
    # We also need to remove the points that were live when the resume file was
    # written, as these show up as samples in both dyn and init
    live_inds = []
    empty_thread_inds = []
    for i, th_lab in enumerate(np.unique(init['thread_labels'])):
        th_inds = np.where(init['thread_labels'] == th_lab)[0]
        live_logl = init['logl'][th_inds[0]]
        init['thread_min_max'][i, 0] = live_logl
        if np.where(dyn['logl'] == live_logl)[0].shape[0] > 0:
            live_inds.append(th_inds[0])
            if th_inds.shape[0] == 1:
                empty_thread_inds.append(i)
        else:
            warnings.warn(
                ('Expected live point at resume should be present in dynamic '
                 'run. If there are no further errors, this warning can be '
                 'ignored.\nlogl={}, th_lab={}, inds={}, init samples (after '
                 'removing first resume_ndead)={}, unique threads in init={}, '
                 'dyn samples={}, resume_ndead={}.').format(
                     str(live_logl), th_lab,
                     np.where(dyn['logl'] == live_logl),
                     init['logl'].shape[0],
                     np.unique(init['thread_labels']).shape[0],
                     dyn['logl'].shape[0], resume_ndead), UserWarning)
    # Remove the live points at resume from init
    init['theta'] = np.delete(init['theta'], live_inds, axis=0)
    for key in ['nlive_array', 'logl', 'thread_labels']:
        init[key] = np.delete(init[key], live_inds)
    # Deal with the case that one of the threads is now empty
    if empty_thread_inds:
        # remove any empty threads from logl_min_max
        init['thread_min_max'] = np.delete(
            init['thread_min_max'], empty_thread_inds, axis=0)
        # Now we need to reorder the thread labels to avoid gaps
        thread_labels_new = np.full(init['thread_labels'].shape, np.nan)
        for i, th_lab in enumerate(np.unique(init['thread_labels'])):
            inds = np.where(init['thread_labels'] == th_lab)[0]
            thread_labels_new[inds] = i
            # Check the newly relabelled thread label matches thread_min_max
            assert init['thread_min_max'][i, 0] <= init['logl'][inds[0]]
            assert init['thread_min_max'][i, 1] == init['logl'][inds[-1]]
        assert np.all(~np.isnan(thread_labels_new))
        init['thread_labels'] = thread_labels_new.astype(int)
    # Add the init threads to dyn with new labels that continue on from the dyn
    # labels
    init['thread_labels'] += dyn['thread_min_max'].shape[0]
    run = nestcheck.ns_run_utils.combine_threads(
        nestcheck.ns_run_utils.get_run_threads(dyn) +
        nestcheck.ns_run_utils.get_run_threads(init),
        assert_birth_point=False)
    return run
