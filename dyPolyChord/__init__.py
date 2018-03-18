#!/usr/bin/env python
"""
__init__.py. Includes wrapper for running dyPolyChord with different
dynamic goals.
"""
import dyPolyChord.run


def run_dypolychord(pc_settings, likelihood, prior, ndims, **kwargs):
    """
    Wrapper for running dynamic polychord with different dynamic goals.
    """
    dynamic_goal = kwargs.pop('dynamic_goal', 1)
    assert dynamic_goal in [0, 1], (
        'dynamic_goal=' + str(dynamic_goal) + '! '
        'So far only set up for dynamic_goal = 0 or 1')
    if dynamic_goal == 1:
        dyPolyChord.run.run_dynamic_polychord_param(
            pc_settings, likelihood, prior, ndims, **kwargs)
    elif dynamic_goal == 0:
        dyPolyChord.run.run_dynamic_polychord_evidence(
            pc_settings, likelihood, prior, ndims, **kwargs)
