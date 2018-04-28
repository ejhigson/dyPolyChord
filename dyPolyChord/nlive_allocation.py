#!/usr/bin/env python
"""
Functions for selecting the number of live points in order to maximise
calculation accuracy.
"""
import warnings
import itertools
import numpy as np
import nestcheck.ns_run_utils
import nestcheck.data_processing


def allocate(init_run, samp_tot, dynamic_goal, smoothing_filter=None):
    """
    Calculates an allocation of life points for dynamic run, checks the output
    allocation and the smoothing applied, and returns the information needed
    for the dynamic run in a dictionary.

    Parameters
    ----------
    init_run: dict
        Initial exploratory run in nestcheck format (see
        http://nestcheck.readthedocs.io/en/latest/api.html for more
        information).
    samp_tot: int
        Total number of samples (including both the initial exploratory run and
        the second dynamic run).
    dynamic_goal: float
    smoothing_filter: function or None, optional
        Smoothing for nlive array. Set to None for no smoothing.

    Returns
    -------
    dyn_info: dict
    """
    assert np.all(np.diff(init_run['logl']) > 0)
    # Calculate nlive allocation with and without smoothing
    nlives = dyn_nlive_array(init_run, samp_tot, dynamic_goal,
                             smoothing_filter=smoothing_filter)
    nlives_unsmoothed = dyn_nlive_array(init_run, samp_tot, dynamic_goal,
                                        smoothing_filter=None)
    # Perform some checks
    if dynamic_goal == 0:
        if not np.all(np.diff(nlives) <= 0):
            assert np.all(np.diff(nlives_unsmoothed) <= 0)
            warnings.warn((
                'Smoothing has added turning points to nlive allocation when '
                'dynamic_goal=0. I am using the unsmoothed nlives '
                'instead.'), UserWarning)
            nlives = nlives_unsmoothed
        assert nlives[0] == nlives.max(), str(nlives)
        assert np.all(np.diff(nlives) <= 0), (
            'When targeting evidence, nlive should monotincally decrease!'
            + ' nlives = ' + str(nlives))
    if dynamic_goal == 1 and nlives[0] != 0:
        warnings.warn(
            'dynamic_goal=1 but the first allocated nlive point is {0}. '
            'For most likelihoods we expect this to equal zero (although '
            'it may be nonzero if there is significant posterior mass '
            'at the edge of the prior).'.format(nlives[0]), UserWarning)
    assert nlives.min() == 0
    # Find number of blocks where nlive is nonzero
    nonzero_blocks = np.asarray(
        [k for k, g in itertools.groupby(nlives != 0)])
    assert nonzero_blocks.sum() == 1, (
        'nlive becomes zero then becomes nonzero! nonzero_blocks='
        + str(nonzero_blocks))
    # Get the indexes of nlives points which are different to the previous
    # points (i.e. remove consecutive duplicates, keeping first occurance)
    inds_to_use = np.concatenate(
        (np.asarray([0]), np.where(np.diff(nlives) != 0)[0] + 1))
    assert (count_turning_points(nlives) ==
            count_turning_points(nlives[inds_to_use]))
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

    Normalisation uses the the formulae for the expected number of samples:

    .. math:: N_\\mathrm{samp} = \\int n(\\log X) \\mathrm{d}\\log X

    where :math:`n` is the local number of live points.

    See Appendix E of "Dynamic nested sampling: an improved algorithm for
    parameter estimation and evidence calculation" (Higson et al., 2018) for
    more information.

    Parameters
    ----------
    init_run: dict
        Initial exploratory run in nestcheck format (see
        http://nestcheck.readthedocs.io/en/latest/api.html for more
        information).
    samp_tot: int
        Total number of samples (including both the initial exploratory run and
        the second dynamic run).
    dynamic_goal: float
    smoothing_filter: function or None, optional
        Smoothing for nlive array. Set to None for no smoothing.

    Returns
    -------
    nlive_array: 1d numpy array
        Number of live points corresponding to each likelihood in
        init_run['logl'].
    """
    assert samp_tot > init_run['logl'].shape[0]
    # Calculate the importance of each point
    importance = sample_importance(init_run, dynamic_goal)
    # Calculate theoretical nlive allocation, which is proportional to
    # importance and normalised to produce an expected samp_tot samples
    logx_init = nestcheck.ns_run_utils.get_logx(init_run['nlive_array'])
    norm = samp_tot / np.abs(np.trapz(importance, x=logx_init))
    importance_nlive = importance * norm
    # Account for the points already sampled
    importance_nlive -= init_run['nlive_array'][0]
    if smoothing_filter is None:
        nlive_array = importance_nlive
    else:
        nlive_array = smoothing_filter(importance_nlive)
    nlive_array = np.clip(nlive_array, 0, None)
    # Renormalise to account for nlives below zero (i.e. regions where we have
    # already taken too many samples) as we cannot take negative samples.
    samp_remain = samp_tot - init_run['logl'].shape[0]
    nlive_array *= samp_remain / np.abs(np.trapz(nlive_array, x=logx_init))
    return np.rint(nlive_array)


def sample_importance(run, dynamic_goal):
    """
    Calculate the importance of each sample in the run.

    See "Dynamic nested sampling: an improved algorithm for parameter
    estimation and evidence calculation" (Higson et al., 2018) for more
    information.

    Parameters
    ----------
    run: dict
        Nested sampling run in nestcheck format (see
        http://nestcheck.readthedocs.io/en/latest/api.html for more
        information)
    dynamic_goal: float or int
        Specifies how much computational effort to put into improving evidence
        and parameter estimation accuracy.

    Returns
    -------
    importance: 1d numpy array
        Importance of each sample. Normalised to np.sum(importance) = 1
    """
    assert 0 <= dynamic_goal <= 1, (
        'dynamic_goal={0} not in [0,1]'.format(dynamic_goal))
    logw = run['logl'] + nestcheck.ns_run_utils.get_logx(run['nlive_array'])
    w_rel = np.exp(logw - logw.max())
    param_imp = w_rel / np.sum(w_rel)
    if dynamic_goal == 1:
        return param_imp
    z_imp = np.cumsum(w_rel)
    z_imp = z_imp.max() - z_imp
    z_imp /= np.sum(z_imp)
    assert z_imp[0] == z_imp.max()
    if dynamic_goal == 0:
        return z_imp
    return (dynamic_goal * param_imp) + ((1 - dynamic_goal) * z_imp)


def count_turning_points(array):
    """Returns number of turning points in a 1d array."""
    # Remove adjacent duplicates only
    uniq = np.asarray([k for k, g in itertools.groupby(array)])
    # Count turning points
    return (np.diff(np.sign(np.diff(uniq))) != 0).sum()
