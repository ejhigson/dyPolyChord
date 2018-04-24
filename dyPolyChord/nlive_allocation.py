#!/usr/bin/env python
"""
Functions for using PyPolyChord, including using it to perform dynamic nested
sampling.
"""
import warnings
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
    # Calculate max number of samples
    if pc_settings_in['max_ndead'] > 0:
        samp_tot = pc_settings_in['max_ndead']
        assert pc_settings_in['max_ndead'] > init_run['logl'].shape[0], (
            'all points used in initial run and none left for dynamic run!')
    else:
        samp_tot = init_run['logl'].shape[0] * (nlive_const / ninit)
        assert nlive_const > ninit
    # Adjust for samples already taken and perform rounding and smoothing
    # operations.
    nlives = dyn_nlive_array(init_run, samp_tot, dynamic_goal,
                             smoothing_filter=smoothing_filter)
    nlives_unsmoothed = dyn_nlive_array(init_run, samp_tot, dynamic_goal,
                                        smoothing_filter=None)
    # ################## Old version ####################################
    # # Calculate number of samples and multiply it by imp to get nlive
    # if pc_settings_in['max_ndead'] > 0:
    #     samp_remain = pc_settings_in['max_ndead'] - init_run['logl'].shape[0]
    #     assert samp_remain > 0, (
    #         'all points used in initial run and none left for dynamic run!')
    # else:
    #     samp_remain = init_run['logl'].shape[0] * ((nlive_const / ninit) - 1)
    # nlives_unsmoothed = imp * samp_remain
    # ###################################################################
    # Get the indexes of nlives points which are different to the previous
    # points (i.e. remove consecutive duplicates, keeping first occurance)
    inds_to_use = np.concatenate(
        (np.asarray([0]), np.where(np.diff(nlives) != 0)[0] + 1))
    # ################################################
    # Perform some checks
    assert np.all(np.diff(init_run['logl']) > 0)
    assert (count_turning_points(nlives) ==
            count_turning_points(nlives[inds_to_use]))
    if dynamic_goal == 0:
        assert nlives[0] > 0
        assert nlives[0] == nlives.max()
        assert np.all(np.diff(nlives) <= 0), (
            'When targeting evidence, nlive should monotincally decrease!'
            + ' nlives = ' + str(nlives))
        assert np.all(np.diff(nlives[inds_to_use]) < 0), (
            'When targeting evidence, nlive should monotincally decrease!'
            + ' nlives = ' + str(nlives))
    elif dynamic_goal == 1:
        if nlives[0] != 0:
            warnings.warn('nlives[0]={0} != 0'.format(nlives[0]), UserWarning)
        # assert nlives[0] == ninit, (
        #     'nlives[0]={0} != ninit={1}'.format(nlives[0], ninit))
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
                'nlives_dict': nlives_dict,
                'peak_start_ind': np.where(nlives > 0)[0][0]}
    return dyn_info


def dyn_nlive_array(init_run, samp_tot, dynamic_goal, smoothing_filter=None):
    """
    Calculate the dynamic nlive allocation from the theoretical, point
    importance-based allocation. This allows for the samples taken in the
    initial run, including areas where more samples than were needed have been
    taken (meaning there are fewer samples available for the regions with peak
    importance). Also perform optional smoothing (carried out *before*
    rounding).

    See Appendix E of "Dynamic nested sampling: an improved algorithm for
    parameter estimation and evidence calculation" (Higson et al., 2018) for
    more information.

    Parameters
    ----------
    """
    # Calculate the importance of each point
    logx_init = nestcheck.ns_run_utils.get_logx(init_run['nlive_array'])
    w_rel = nestcheck.plots.rel_posterior_mass(logx_init, init_run['logl'])
    if dynamic_goal == 0:
        imp = np.cumsum(w_rel)
        imp = imp.max() - imp
        assert imp[0] == imp.max()
    elif dynamic_goal == 1:
        imp = w_rel
    imp /= np.abs(np.trapz(imp, x=logx_init))
    assert imp.min() >= 0
    # Calculate theoretical nlive allocation based on imporance and samp_tot
    # only
    importance_nlive = imp * samp_tot
    # Account for the points already sampled
    nlive_array = importance_nlive - init_run['nlive_array']
    if smoothing_filter is not None:
        nlive_array = smoothing_filter(nlive_array)
    nlive_array = np.clip(nlive_array, 0, None)
    # Renormalise to account for nlives below zero (i.e. regions where we have
    # already taken too many samples) as we cannot take negative samples.
    samp_remain = samp_tot - init_run['logl'].shape[0]
    nlive_array *= (samp_remain / np.abs(np.trapz(nlive_array, x=logx_init)))
    return np.rint(nlive_array)


def count_turning_points(array):
    """Returns number of turning points in a 1d array."""
    # remove adjacent duplicates only
    uniq = np.asarray([k for k, g in itertools.groupby(array)])
    return (np.diff(np.sign(np.diff(uniq))) != 0).sum()
