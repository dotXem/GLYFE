import numpy as np


def RMSE(results):
    """
    Compute the Root-Mean-Squared Error of the predictions.
    :param results: dataframe with predictions and ground truths
    :return: sqrt(1/n * sum((y_true - y_pred)^2))
    """
    y_true, y_pred = results.values.transpose()

    return np.sqrt(np.nanmean((y_true - y_pred) ** 2))
