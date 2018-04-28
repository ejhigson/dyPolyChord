#!/usr/bin/env python
"""
Run dyPolyChord using compiled Fortran or C++ likelihoods.
"""
import functools
# import copy
import os


def get_compiled_run_func(ex_path, ndim, prior_name, prior_params,
                          derived_str=None):
    prior_block_str = get_prior_block_str(prior_name, prior_params, ndim)
    return functools.partial(compiled_run_func, ex_path=ex_path,
                             prior_block_str=prior_block_str,
                             derived_str=derived_str)


def compiled_run_func(settings_dict, prior_block_str=None, ex_path=None,
                      derived_str=None):
    """python_run_func."""
    ini_path = os.path.join(settings_dict['base_dir'],
                            settings_dict['file_root'] + '.ini')
    write_ini(settings_dict, prior_block_str, ini_path,
              derived_str=derived_str)
    os.system(ex_path + ' ' + ini_path)


def write_ini(settings, prior_block_str, file_path, derived_str=None):
    """Write an ini file based on the settings."""
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


def get_prior_block_str(name, prior_params, ndim, speed=1, block=1):
    block_str = ''
    for i in range(1, ndim + 1):
        block_str += ('P : p{0} | \\theta_{{{0}}} | {1} | {2} | {3} |'
                      .format(i, speed, name, block))
        block_str += format_setting(list(prior_params)) + '\n'
    return block_str


def format_setting(setting):
    """Return setting as string using 'T' for True and 'F' for False."""
    if isinstance(setting, bool):
        return str(setting)[0]
    elif isinstance(setting, list):
        return str(setting).replace('[', '').replace(']', '').replace(',', '')
    else:
        return str(setting)
