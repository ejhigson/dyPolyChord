#!/usr/bin/env python
"""
__init__.py. Includes wrapper for running dyPolyChord with different
dynamic goals.
"""
import dyPolyChord.run_dynamic_ns


def run_dypolychord(pc_settings, likelihood, prior, ndims, **kwargs):
    """
    Wrapper for running dynamic polychord with different dynamic goals.
    """
    dynamic_goal = kwargs.pop('dynamic_goal', 1)
    assert dynamic_goal in [0, 1], (
        'dynamic_goal=' + str(dynamic_goal) + '! '
        'So far only set up for dynamic_goal = 0 or 1')
    dyPolyChord.run_dynamic_ns.run_dypolychord(
        pc_settings, dynamic_goal, likelihood, prior, ndims, **kwargs)
