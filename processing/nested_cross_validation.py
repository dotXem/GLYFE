import misc.constants as cs
import numpy as np
from tuning.coarse_to_fine_tuning import compute_coarse_params_grid, compute_refined_params_grid
from evaluation.rmse import RMSE


def nested_cross_validation(subject, model_class, ph, params, search, train, valid, test):
    """
    Nested cross-validation: a first (inner) cross-validation (factor cv-1) is done to train and tune the model (on the
    training and validation set) and a second (outer) cross-validation is done to do the final predictions on the test set.
    :param subject: subject on which is trained the personalized model (string)
    :param model_class: name of the model (class)
    :param params: parameters for the model (dict)
    :param search: coarse and refine instructions for the parameters for the automated tuning (dict)
    :param train: training set (array of DataFrame)
    :param valid: validation set (array of DataFrame)
    :param test: testing set (array of DataFrame)
    :return: array containing the (true,pred) tuples for every fold
    """
    results = []
    for outer_loop_index in range(cs.cv):
        # if there is a automated tuning of the hyperparameters...
        if search:
            # compute coarse grid search parameters and run the search on the inner loop
            coarse_params_grid = compute_coarse_params_grid(params, search)

            # define a convenience function for params search, returns the best params
            def params_search(grid):
                results = []
                for params_tmp in grid:
                    results_tmp = []
                    for inner_loop_index in range(cs.cv - 1):
                        index = outer_loop_index * (cs.cv - 1) + inner_loop_index
                        model = model_class(subject, ph, params_tmp, train[index], valid[index], test[index])
                        model.fit()
                        y_true, y_pred = model.predict(dataset="valid")
                        results_tmp.append(RMSE(y_true, y_pred))
                    results.append(results_tmp)
                return grid[np.argmin(np.mean(np.transpose(results),axis=0))]

            # compute the best coarse params on the inner loop
            best_coarse_params = params_search(coarse_params_grid)

            # compute refinement grid search parameters
            refined_params_grid = compute_refined_params_grid(params, search, best_coarse_params)

            # compute the best refined params on the inner loop
            best_refined_params = params_search(refined_params_grid)
        else:
            best_refined_params = params

        # run the inner loop as last time with the best parameters to compute the final results
        for inner_loop_index in range(cs.cv - 1):
            index = outer_loop_index * (cs.cv - 1) + inner_loop_index
            model = model_class(subject, ph, best_refined_params, train[index], valid[index], test[index])
            model.fit()
            y_true, y_pred = model.predict(dataset="test")
            results.append(np.c_[y_true.reshape(-1, 1), y_pred.reshape(-1, 1)])

    return np.rollaxis(np.array(results),2)
