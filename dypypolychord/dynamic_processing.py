#!/usr/bin/env python
"""
Processing dynamic runs.
"""
import numpy as np
import nestcheck.analyse_run as ar
import nestcheck.data_processing


def process_dyn_run(root):
    """
    Process dynamic nested sampling run including both initial exploratory run
    and second dynamic run.
    """
    init = nestcheck.data_processing.process_polychord_run(root + '_init')
    dyn = nestcheck.data_processing.process_polychord_run(root + '_dyn')
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
