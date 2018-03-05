#!/usr/bin/env python
"""
Helper functions for saving and loading PolyChord data produced with
dypypolychord.
"""
import nestcheck.io_utils as iou
import dypypolychord.dynamic_processing


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


def settings_root(likelihood_name, prior_name, ndims, **kwargs):
    """Get a standard string containing information about settings."""
    dynamic_goal = kwargs.pop('dynamic_goal')
    nlive_const = kwargs.pop('nlive_const')
    nrepeats = kwargs.pop('nrepeats')
    ninit = kwargs.pop('ninit', None)
    dyn_nlive_step = kwargs.pop('dyn_nlive_step', None)
    init_step = kwargs.pop('init_step', None)
    root = likelihood_name + '_' + prior_name + '_dg' + str(dynamic_goal)
    if dynamic_goal is not None:
        assert ninit is not None
        assert dyn_nlive_step is not None
        root += '_' + str(ninit) + 'init_' + str(dyn_nlive_step) + 'ds'
        if dynamic_goal != 0:
            assert init_step is not None
            root += '_' + str(init_step) + 'is'
    root += '_' + str(ndims) + 'd'
    root += '_' + str(nlive_const) + 'nlive'
    root += '_' + str(nrepeats) + 'nrepeats'
    return root


def get_dypypolychord_data(file_root, n_runs, dynamic_goal, **kwargs):
    """
    Load and process polychord chains
    """
    data_dir = kwargs.pop('data_dir', 'cache/')
    chains_dir = kwargs.pop('chains_dir', 'chains/')
    load = kwargs.pop('load', False)
    save = kwargs.pop('save', False)
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
            data.append(dypypolychord.dynamic_processing
                        .process_dypypolychord_run(root, dynamic_goal))
        except (OSError, AssertionError, KeyError) as err:
            try:
                errors[type(err).__name__].append(i)
            except KeyError:
                errors[type(err).__name__] = [i]
    for error_name, val_list in errors.items():
        if val_list:
            save = False  # only save if every file is processed ok
            message = (error_name + ' processing ' + str(len(val_list)) + ' / '
                       + str(n_runs) + ' files')
            if len(val_list) != n_runs:
                message += '. Runs with errors were: ' + str(val_list)
            print(message)
    if save:
        print('Processed new chains: saving to ' + save_name)
        iou.pickle_save(data, data_dir + save_name, print_time=False,
                        overwrite_existing=overwrite_existing)
    return data
