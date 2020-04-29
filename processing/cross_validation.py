import numpy as np
from misc.utils import printd
from postprocessing.metrics.rmse import RMSE
from processing.hyperparameters_tuning import compute_coarse_params_grid, compute_refined_params_grid


def make_predictions(subject, model_class, params, ph, train, valid, test, mode="valid"):
    """
    For every train, valid, test fold, fit the given model with params at prediciton horizon on the training set,
    and make predictions on either the validation or testing set
    :param subject: name of subject
    :param model_class: name of model
    :param params: hyperparameters file name
    :param ph: prediction horizon in scaled minutes
    :param train: training sets
    :param valid: validation sets
    :param test: testing sets
    :param mode: on which set the model is tested (either "valid" or "test")
    :return: array of ground truths/predictions dataframe
    """
    results = []
    for train_i, valid_i, test_i in zip(train, valid, test):
        model = model_class(subject, ph, params, train_i, valid_i, test_i)
        model.fit()
        res = model.predict(dataset=mode)
        results.append(res)
        if mode == "valid":
            break
    return results


def find_best_hyperparameters(subject, model_class, params, search, ph, train, valid, test):
    coarse_params_grid = compute_coarse_params_grid(params, search)
    def params_search(grid):
        results = []
        for params_tmp in grid:
            res = make_predictions(subject,model_class,params_tmp, ph, train, valid, test, mode="valid")
            results.append([RMSE(res_) for res_ in res])
            printd(params_tmp, results[-1])
        return grid[np.argmin(np.mean(np.transpose(results), axis=0))]

    # compute the best coarse params on the inner loop
    best_coarse_params = params_search(coarse_params_grid)

    # compute refinement grid search parameters
    refined_params_grid = compute_refined_params_grid(params, search, best_coarse_params)

    # compute the best refined params on the inner loop
    best_refined_params = params_search(refined_params_grid)

    return best_refined_params
