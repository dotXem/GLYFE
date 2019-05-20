"""
This file explains how to build a params file for the models. For example, one can look at the other params files inside
the directory.
"""

parameters = {
    """
    Dictionary that defines the parameters of the model. While a fixed hyperparameter is presented by a (key,val) tuple.
    A hyperparameter which needs to be optimized is represented by a (key, [lb, ub]) tuple; lb and ub being respectively
    the lower and upper bound of the search space.
    """
}

search = {
    """
    Defines the searching metholody for the hyperparameters that need to be optimized. For every such field, we should 
    have a (key, [method, n_coarse, n_refine]) tuple. n_coarse represents the number of coarse values to be tested inside
    the initial search space; n_refine represents the number of refined values to be tested around the best coarse value;
    method should be either "linear" or "logarithmic" and represents the search methodology inside the initial space.
    """
}
