#!/usr/bin/env python
"""
Functions for running dyPolyChord using compiled PolyChord C++ or Fortran
likelihoods.
"""
import functools
import os


def get_compiled_run_func(ex_path, prior_block_str, derived_str=None):
    """
    Helper function for freezing python_run_func args.

    Parameters
    ----------
    ex_path: str
    prior_block_str: str
    derived_str: str or None, optional

    Returns
    -------
    functools partial object
        compiled_run_func with input parameters frozen.
    """
    return functools.partial(compiled_run_func, ex_path=ex_path,
                             prior_block_str=prior_block_str,
                             derived_str=derived_str)


def compiled_run_func(settings_dict, prior_block_str=None, ex_path=None,
                      derived_str=None):
    """
    Runs a PolyChord executable with the specified inputs by writing a .ini
    file containing the input settings.
    See the PolyChord documentation for more details.

    Parameters
    ----------
    settings_dict: dict
        Input PolyChord settings.
    prior_block_str: str
        String specifying prior parameters (see get_prior_block_str for more
        details).
    exp_path: str
        String containing path to compiled PolyChord likelihood.
    derived_str: str or None, optional
    """
    ini_path = os.path.join(settings_dict['base_dir'],
                            settings_dict['file_root'] + '.ini')
    write_ini(settings_dict, prior_block_str, ini_path,
              derived_str=derived_str)
    os.system(ex_path + ' ' + ini_path)


def write_ini(settings, prior_block_str, file_path, derived_str=None):
    """
    Writes a PolyChord format .ini file based on the input settings.

    Parameters
    ----------
    settings_dict: dict
        Input PolyChord settings.
    prior_block_str: str
        String specifying prior parameters (see get_prior_block_str for more
        details).
    file_path: str
        Path to write ini file to.
    derived_str: str or None, optional
    """
    with open(file_path, 'w') as ini_file:
        # Write the settings
        for key, value in settings.items():
            if key == 'nlives':
                if value:
                    loglikes = sorted(settings['nlives'])
                    ini_file.write(('loglikes = ' + format_setting(loglikes)
                                    + '\n'))
                    nlives = [settings['nlives'][logl] for logl in loglikes]
                    ini_file.write('nlives = ' + format_setting(nlives) + '\n')
            else:
                ini_file.write(key + ' = ' + format_setting(value) + '\n')
        # write the prior
        ini_file.write(prior_block_str)
        if derived_str is not None:
            ini_file.write(derived_str)


def get_prior_block_str(prior_name, prior_params, nparam, **kwargs):
    """
    Returns a PolyChord format prior block for inclusion in PolyChord .ini
    files.

    See the PolyChord documentation for more details.

    Parameters
    ----------
    prior_name: str
        Name of prior. See the PolyChord documnetation for a list of currently
        available priors and details of how to add your own.
    prior_params: str, float or list of strs and floats
        Parameters for the prior function.
    nparam: int
        Number of parameters.
    start_param: int, optional
        Where to start param numbering. For when multiple prior blocks are
        being used.
    block: int, optional
        Number of block (only needed when using multiple prior blocks).
    speed: int, optional
        Use to specify fast and slow parameters if required. See the PolyChord
        documnetation for more details.

    Returns
    -------
    block_str: str
        PolyChord format prior block.
    """
    start_param = kwargs.pop('start_param', 1)
    speed = kwargs.pop('speed', 1)
    block = kwargs.pop('block', 1)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    block_str = ''
    for i in range(start_param, nparam + start_param):
        block_str += ('P : p{0} | \\theta_{{{0}}} | {1} | {2} | {3} |'
                      .format(i, speed, prior_name, block))
        block_str += format_setting(prior_params) + '\n'
    return block_str


def format_setting(setting):
    """
    Return setting as string in the format needed for PolyChord's .ini files.
    These use 'T' for True and 'F' for False, and require lists of numbers
    written separated by spaces and without commas or brackets.

    Parameters
    ----------
    prior_name: str
        Name of prior. See the PolyChord documnetation for a list of currently
        available priors and details of how to add your own.
    prior_params: str, float or list of strs and floats
        Parameters for the prior block.
    nparam: int
        Number of parameters.
    start_param: int, optional
        Where to start param numbering. For when multiple prior blocks are
        being used.
    block: int, optional
        Number of block (only needed when using multiple prior blocks).
    speed: int, optional
        Use to specify fast and slow parameters if required. See the PolyChord
        documnetation for more details.

    Returns
    -------
    str
    """
    if isinstance(setting, bool):
        return str(setting)[0]
    elif isinstance(setting, (list, tuple)):
        string = str(setting)
        for char in [',', '[', ']', '(', ')']:
            string = string.replace(char, '')
        return string
    else:
        return str(setting)
