import numpy as np


def RMSE(y_true, y_pred):
    """
    Compute the Root Mean Square Error of the predictions.
    :param y_true: true values;
    :param y_pred: predicted values;
    :return: sqrt(1/n * sum((y_true - y_pred)^2))
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
