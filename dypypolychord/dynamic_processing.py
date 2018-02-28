#!/usr/bin/env python
"""
Processing dynamic runs.
"""
import numpy as np
import nestcheck.analyse_run as ar
import nestcheck.data_processing


def process_dypypolychord_run(root, dynamic_goal):
    assert dynamic_goal in [None, 0, 1], (
        'dynamic_goal=' + str(dynamic_goal) + '! '
        'So far only set up for dynamic_goal = None, 0, 1')
    if dynamic_goal is None:
        return nestcheck.data_processing.process_polychord_run(root)
    elif dynamic_goal == 0:
        init = nestcheck.data_processing.process_polychord_run(root + '_init')
        assert np.all(init['thread_min_max'][:, 0] == -np.inf)
        dyn = nestcheck.data_processing.process_polychord_run(root + '_dyn')
        run = ar.combine_ns_runs([init, dyn])
        nestcheck.data_processing.check_ns_run(run)
        return run
    elif dynamic_goal == 1:
        return process_dyn_run(root)


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
        assert np.where(dyn['logl'] == live_logl)[0].shape == (1,), \
            'this point should be present in dyn too!'
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
    run['settings'] = {'resume_ndead': resume_ndead}
    try:
        nestcheck.data_processing.check_ns_run(run)
    except Exception as err:
        # Add info on resume_ndead to the error message
        import sys
        raise (type(err)(str(err) + '. resume_ndead=%s' % resume_ndead)
               .with_traceback(sys.exc_info()[2]))
    return run
