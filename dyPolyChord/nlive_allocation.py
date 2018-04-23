#!/usr/bin/env python
"""
Functions for using PyPolyChord, including using it to perform dynamic nested
sampling.
"""
import itertools
import numpy as np
import scipy.signal
import nestcheck.plots
import nestcheck.ns_run_utils
import nestcheck.data_processing


def allocate(pc_settings_in, ninit, nlive_const, dynamic_goal, **kwargs):
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
        pc_settings_in['file_root'] + '_init', pc_settings_in['base_dir'])
    logx_init = nestcheck.ns_run_utils.get_logx(init_run['nlive_array'])
    w_rel = nestcheck.plots.rel_posterior_mass(logx_init, init_run['logl'])
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
    if pc_settings_in['max_ndead'] > 0:
        samp_remain = pc_settings_in['max_ndead'] - init_run['logl'].shape[0]
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
