#!/usr/bin/env python
"""
Functions for running dyPolyChord using compiled PolyChord C++ or Fortran
likelihoods.
"""
import os


def get_prior_block_str(prior_name, prior_params, nparam, **kwargs):
    """
    Returns a PolyChord format prior block for inclusion in PolyChord .ini
    files.

    See the PolyChord documentation for more details.

    Parameters
    ----------
    prior_name: str
        Name of prior. See the PolyChord documentation for a list of currently
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
        documentation for more details.

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


class RunCompiledPolyChord(object):

    """Object for running a compiled PolyChord executable with specified
    inputs."""

    def __init__(self, ex_path, prior_str, derived_str=None):
        """
        Specify path to executable, priors and derived parameters.

        Parameters
        ----------
        ex_path: str
        prior_str: str
            String specifying prior in the format required for PolyChord .ini
            files (see prior_str for more details).
        derived_str: str or None, optional
        """
        self.ex_path = ex_path
        self.prior_str = prior_str
        self.derived_str = derived_str

    def __call__(self, settings_dict, comm=None):
        """
        Run PolyChord with the input settings by writing a .ini file then using
        the compiled likelihood specified in ex_path.

        See the PolyChord documentation for more details.

        Parameters
        ----------
        settings_dict: dict
            Input PolyChord settings.
        comm: None, optional
            Not used. Included only so __call__ has the same arguments as the
            equivalent python function (which uses the comm argument for
            runnign with MPI).
        """
        assert comm is None
        ini_path = os.path.join(settings_dict['base_dir'],
                                settings_dict['file_root'] + '.ini')
        self.write_ini(settings_dict, ini_path)
        os.system(self.ex_path + ' ' + ini_path)

    def write_ini(self, settings, file_path):
        """
        Writes a PolyChord format .ini file based on the input settings.

        Parameters
        ----------
        settings: dict
            Input PolyChord settings.
        file_path: str
            Path to write ini file to.
        """
        with open(file_path, 'w') as ini_file:
            # Write the settings
            for key, value in settings.items():
                if key == 'nlives':
                    if value:
                        loglikes = sorted(settings['nlives'])
                        ini_file.write(
                            'loglikes = ' + format_setting(loglikes) + '\n')
                        nlives = [settings['nlives'][ll] for ll in loglikes]
                        ini_file.write(
                            'nlives = ' + format_setting(nlives) + '\n')
                else:
                    ini_file.write(key + ' = ' + format_setting(value) + '\n')
            # write the prior
            ini_file.write(self.prior_str)
            if self.derived_str is not None:
                ini_file.write(self.derived_str)


def format_setting(setting):
    """
    Return setting as string in the format needed for PolyChord's .ini files.
    These use 'T' for True and 'F' for False, and require lists of numbers
    written separated by spaces and without commas or brackets.

    Parameters
    ----------
    setting: (can be any type for which str(settings) works)

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
