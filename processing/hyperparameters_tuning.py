import sys

import numpy as np

from misc.utils import dict_cartesian_product


def compute_coarse_params_grid(params, search):
    """
    Compute a list of params based on the params boundaries and search rules.
    :param params: params dict;
    :param search: seach dict;
    :return: list of params dict;
    """
    # reformat the params
    params_ = params.copy()
    for key, val in params_.items():
        if key in search.keys():
            method, num_coarse, _ = search[key]
            if method == "logarithmic":
                [start, stop] = val
                space = np.logspace(np.log10(start), np.log10(stop), num=num_coarse, endpoint=True, base=10)
            elif method == "linear":
                [start, stop] = val
                space = np.linspace(start, stop, num=num_coarse, endpoint=True)
            elif method == "list":
                space = val
            else:
                sys.exit(-1)
            params_[key] = space
        else:
            params_[key] = [val]

    # compute the cartesian product of the dict parameters
    combinations = dict_cartesian_product(params_)

    params_grid = [{x: vals[i] for i, x in enumerate(params_.copy())} for vals in combinations]

    return params_grid


def compute_refined_params_grid(params, search, best_params):
    """
    Compute a list of params based on the params boundaries and search rules.
    :param params: params dict;
    :param search: seach dict;
    :param best_params: best hyperparam combination from coarse search
    :return: list of params dict
    """
    best_params_ = best_params.copy()
    for key, val in best_params_.items():
        if key in search:
            method, num_coarse, num_refined = search[key]

            if method == "logarithmic":
                step = np.ceil((params[key][-1] / params[key][0]) ** (1 / (num_coarse)))
                mod = [step ** i for i in (np.arange(num_refined) - num_refined // 2) / (num_refined // 2 + 1)]
                best_params_[key] = [val * i for i in mod]
            elif method == "linear":
                step = np.ceil((params[key][-1] - params[key][0]) / ((num_coarse - 1) * num_refined))
                mod = step * (np.arange(num_refined) - num_refined // 2)
                best_params_[key] = [val + i for i in mod]
            elif method == "list":
                best_params_[key] = [val]
            else:
                sys.exit(-1)
        else:
            best_params_[key] = [val]

    # compute the cartesian product of the dict parameters
    combinations = dict_cartesian_product(best_params_)

    params_grid = [{x: vals[i] for i, x in enumerate(best_params_.copy())} for vals in combinations]

    return params_grid