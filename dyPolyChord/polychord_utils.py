#!/usr/bin/env python
"""
Functions for running dyPolyChord using compiled PolyChord C++ or Fortran
likelihoods.
"""
import os


class RunCompiledPolyChord(object):

    """Object for running a compiled PolyChord executable with specified
    inputs."""

    def __init__(self, executable_path, prior_str, **kwargs):
        """
        Specify path to executable, priors and derived parameters.

        Parameters
        ----------
        executable_path: str
            Path to compiled likelihood. If this is in the directory from which
            dyPolyChord is being run, you may need to prepend "./" to the
            executable name for it to work.
        prior_str: str
            String specifying prior in the format required for PolyChord .ini
            files (see get_prior_block_str for more details).
        config_str: str, optional
            String to be written to [root].cfg file if required.
        derived_str: str or None, optional
            String specifying prior in the format required for PolyChord .ini
            files (see prior_str for more details).
        mpi_str: str or None, optional
            Optional mpi command to preprend to run command.
            For example to run with 8 processors, use mpi_str = 'mprun -np 8'.
            Note that PolyChord must be installed with MPI enabled to allow
            running with MPI.
        """
        self.config_str = kwargs.pop('config_str', None)
        self.derived_str = kwargs.pop('derived_str', None)
        self.mpi_str = kwargs.pop('mpi_str', None)
        if kwargs:
            raise TypeError('unexpected **kwargs: {0}'.format(kwargs))
        self.executable_path = executable_path
        self.prior_str = prior_str

    def __call__(self, settings_dict, comm=None):
        """
        Run PolyChord with the input settings by writing a .ini file then using
        the compiled likelihood specified in executable_path.

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
        assert os.path.isfile(self.executable_path), (
            'executable not found: ' + self.executable_path)
        assert comm is None, 'comm not used for compiled likelihoods.'
        # Write settings to ini file
        file_path = os.path.join(
            settings_dict['base_dir'], settings_dict['file_root'])
        with open(file_path + '.ini', 'w') as ini_file:
            ini_file.write(self.ini_string(settings_dict))
        # If required, write config file
        if self.config_str is not None:
            with open(file_path + '.cfg', 'w') as cfg_file:
                cfg_file.write(self.config_str)
        # Execute command
        command_str = self.executable_path + ' ' + file_path + '.ini'
        if self.mpi_str is not None:
            command_str = self.mpi_str + ' ' + command_str
        os.system(command_str)

    def ini_string(self, settings):
        """Get a PolyChord format .ini file string based on the input settings.
        """
        string = ''
        # Add the settings
        for key, value in settings.items():
            if key == 'nlives':
                if value:
                    loglikes = sorted(settings['nlives'])
                    string += 'loglikes = ' + format_setting(loglikes) + '\n'
                    nlives = [settings['nlives'][ll] for ll in loglikes]
                    string += 'nlives = ' + format_setting(nlives) + '\n'
            else:
                string += key + ' = ' + format_setting(value) + '\n'
        # Add the prior
        string += self.prior_str
        if self.derived_str is not None:
            string += self.derived_str
        return string


# Helper functions for making PolyChord prior strings
# ---------------------------------------------------


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
        PolyChord format prior block string for ini file.
    """
    start_param = kwargs.pop('start_param', 1)
    speed = kwargs.pop('speed', 1)
    block = kwargs.pop('block', 1)
    if kwargs:
        raise TypeError('unexpected **kwargs: {0}'.format(kwargs))
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


def python_prior_to_str(prior, **kwargs):
    """Utility function for mapping python priors (of the type in
    python_priors.py) to  ini file format strings used for compiled (C++
    or Fortran) likelihoods.

    The input prior must correspond to a prior function set up in
    PolyChord/src/polychord/priors.f90. You can easily add your own too.
    Note that some of the priors are only available in PolyChord >= v1.15.

    Parameters
    ----------
    prior_obj: python prior object
        Of the type defined in python_priors.py
    kwargs: dict, optional
        Passed to get_prior_block_str (see its docstring for more details).

    Returns
    -------
    block_str: str
        PolyChord format prior block string for ini file.
    """
    nparam = kwargs.pop('nparam')
    name = type(prior).__name__.lower()
    if name == 'uniform':
        parameters = [prior.minimum, prior.maximum]
    elif name == 'poweruniform':
        name = 'power_uniform'
        parameters = [prior.minimum, prior.maximum, prior.power]
        assert prior.power < 0, (
            'compiled power_uniform currently only takes negative powers.'
            'power={}'.format(prior.power))
    elif name == 'gaussian':
        parameters = [getattr(prior, 'mu', 0.0), prior.sigma]
        if getattr(prior, 'half', False):
            name = 'half_' + name
    elif name == 'exponential':
        parameters = [prior.lambd]
    else:
        raise TypeError('Not set up for ' + name)
    if getattr(prior, 'sort', False):
        name = 'sorted_' + name
    if getattr(prior, 'adaptive', False):
        name = 'adaptive_' + name
        assert getattr(prior, "nfunc_min", 1) == 1, (
            'compiled adaptive priors currently only take nfunc_min=1.'
            'prior.nfunc_min={}'.format(prior.nfunc_min))
    return get_prior_block_str(name, parameters, nparam, **kwargs)


def python_block_prior_to_str(bp_obj):
    """As for python_prior_to_str, but for BlockPrior objects of the type
    defined in python_priors.py. python_prior_to_str is called seperately on
    every block.

    Parameters
    ----------
    prior_obj: python prior object
        Of the type defined in python_priors.py.
    kwargs: dict, optional
        Passed to get_prior_block_str (see its docstring for more details).

    Returns
    -------
    block_str: str
        PolyChord format prior block string for ini file.
    """
    assert type(bp_obj).__name__ == 'BlockPrior', (
        'Unexpected input object type: {}'.format(
            type(bp_obj).__name__))
    start_param = 1
    string = ''
    for i, prior in enumerate(bp_obj.prior_blocks):
        string += python_prior_to_str(
            prior, block=(i + 1), start_param=start_param,
            nparam=bp_obj.block_sizes[i])
        start_param += bp_obj.block_sizes[i]
    return string
