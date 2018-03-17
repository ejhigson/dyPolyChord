#!/usr/bin/env python
"""
__init__. Also includes wrapper for running dyPyPolyChord with different
dynamic goals.
"""
import dypypolychord.run


def run_dypypolychord(pc_settings, likelihood, prior, ndims, **kwargs):
    """
    Wrapper for running dynamic polychord with different dynamic goals.
    """
    dynamic_goal = kwargs.pop('dynamic_goal', 1)
    assert dynamic_goal in [None, 0, 1], (
        'dynamic_goal=' + str(dynamic_goal) + '! '
        'So far only set up for dynamic_goal = None, 0, 1')
    if dynamic_goal == 1:
        dypypolychord.run.run_dynamic_polychord_param(
            pc_settings, likelihood, prior, ndims, **kwargs)
    elif dynamic_goal == 0:
        dypypolychord.run.run_dynamic_polychord_evidence(
            pc_settings, likelihood, prior, ndims, **kwargs)
    elif dynamic_goal is None:
        dypypolychord.run.run_standard_polychord(
            pc_settings, likelihood, prior, ndims, **kwargs)
