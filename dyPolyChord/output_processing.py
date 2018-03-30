#!/usr/bin/env python
"""
Functions for loading and processing dynamic runs.
"""
import numpy as np
import nestcheck.analyse_run as ar
import nestcheck.data_processing
import nestcheck.io_utils as iou


def settings_root(likelihood_name, prior_name, ndims, **kwargs):
    """Get a standard string containing information about settings."""
    prior_scale = kwargs.pop('prior_scale')
    dynamic_goal = kwargs.pop('dynamic_goal')
    nlive_const = kwargs.pop('nlive_const')
    nrepeats = kwargs.pop('nrepeats')
    ninit = kwargs.pop('ninit', None)
    init_step = kwargs.pop('init_step', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    root = likelihood_name + '_' + prior_name + '_' + str(prior_scale)
    root += '_dg' + str(dynamic_goal)
    if dynamic_goal is not None:
        assert ninit is not None
        root += '_' + str(ninit) + 'init'
        if dynamic_goal != 0:
            assert init_step is not None
            root += '_' + str(init_step) + 'is'
    root += '_' + str(ndims) + 'd'
    root += '_' + str(nlive_const) + 'nlive'
    root += '_' + str(nrepeats) + 'nrepeats'
    return root


def process_dypolychord_run(file_root, base_dir, **kwargs):
    """
    Load the output files of a dynamic run and process them to the nestcheck
    format.
    """
    dynamic_goal = kwargs.pop('dynamic_goal')
    logl_warn_only = kwargs.pop('logl_warn_only', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert dynamic_goal in [0, 1], (
        'dynamic_goal=' + str(dynamic_goal) + '! '
        'So far only set up for dynamic_goal = 0 or 1')
    init = nestcheck.data_processing.process_polychord_run(
        file_root + '_init', base_dir, logl_warn_only=logl_warn_only)
    dyn = nestcheck.data_processing.process_polychord_run(
        file_root + '_dyn', base_dir, logl_warn_only=logl_warn_only)
    assert np.all(init['thread_min_max'][:, 0] == -np.inf), (
        str(init['thread_min_max']))
    if dynamic_goal == 0:
        # If dynamic_goal == 0, dyn was not resumed part way through init
        # and we can simply combine dyn and init using standard nestcheck
        # functions
        run = ar.combine_ns_runs([init, dyn])
        run['output'] = {'nlike': (init['output']['nlike'] +
                                   dyn['output']['nlike'])}
        # assert np.all(run['thread_min_max'][:, 0] == -np.inf), (
        if not np.all(run['thread_min_max'][:, 0] == -np.inf):
            print(
                str((run['thread_min_max'][:, 0] != -np.inf).sum()) + ' / ' +
                str(run['thread_min_max'].shape[0]) + ' threads dont start at ' +
                '-np.inf. They have thread_min_max values: ' +
                str(run['thread_min_max']
                    [np.where(run['thread_min_max'][:, 0] != -np.inf)[0], :]))
    elif dynamic_goal == 1:
        # If dynamic_goal == 1, dyn was resumed part way through init and we
        # need to remove duplicate points from the combined run
        dyn_info = iou.pickle_load(base_dir + '/' + file_root + '_dyn_info')
        run = combine_resumed_dyn_run(init, dyn, dyn_info['resume_ndead'])
        run['output'] = dyn_info
        run['output']['nlike'] = (init['output']['nlike'] +
                                  dyn['output']['nlike'] -
                                  dyn_info['resume_nlike'])
    run['output']['file_root'] = file_root
    run['output']['base_dir'] = base_dir
    run['output']['dynamic_goal'] = dynamic_goal
    run['output']['init_nlike'] = init['output']['nlike']
    run['output']['dyn_nlike'] = dyn['output']['nlike']
    # check the nested sampling run has the expected properties and resume
    nestcheck.data_processing.check_ns_run(run)
    return run


def combine_resumed_dyn_run(init, dyn, resume_ndead):
    """
    Process dynamic nested sampling run including both initial exploratory run
    and second dynamic run.
    """
    # Remove the first resume_ndead points which are in both dyn and init from
    # init
    assert np.array_equal(init['logl'][:resume_ndead],
                          dyn['logl'][:resume_ndead])
    init['theta'] = init['theta'][resume_ndead:, :]
    for key in ['nlive_array', 'logl', 'thread_labels']:
        init[key] = init[key][resume_ndead:]
    # We also need to remove the points that were live when the resume file was
    # written, as these show up as dead points in dyn
    live_inds = []
    empty_thread_inds = []
    for i, th_lab in enumerate(np.unique(init['thread_labels'])):
        th_inds = np.where(init['thread_labels'] == th_lab)[0]
        live_inds.append(th_inds[0])
        live_logl = init['logl'][th_inds[0]]
        if th_inds.shape[0] == 1:
            empty_thread_inds.append(i)
        assert np.where(dyn['logl'] == live_logl)[0].shape == (1,), (
            'point should be present in dyn too! logl=' + str(live_logl))
        init['thread_min_max'][i, 0] = live_logl
    # lets remove the live points at init
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
    run = ar.combine_threads(ar.get_run_threads(dyn) +
                             ar.get_run_threads(init),
                             assert_birth_point=True)
    return run
