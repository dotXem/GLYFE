import numpy as np

def MAPE(results):
    """
        Compute the mean absolute percentage error  of the predictions, with is a normalized mean absolute error, expressed in %
        :param results: dataframe with predictions and ground truths
        :return: fitness
    """
    y_true, y_pred = results.values.transpose()
    return 100 * np.nanmean(np.abs((y_true-y_pred)/y_true))