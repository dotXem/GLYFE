import numpy as np


def MASE(results, ph, freq):
    """
        Compute the mean absolute scaled error, with is a the MAE noramized by the MAE of a "naïve" prediction
        (value has not change between the time t and t+PH)
        :param results: dataframe with predictions and ground truths
        :param freq: sampling frequency in minutes
        :return: fitness
    """
    ph_f = ph // freq
    y_true, y_pred = results.values.transpose()
    num = np.nanmean(np.abs(y_true - y_pred))
    denom = np.nanmean(np.abs(y_true[:-ph_f] - y_true[ph_f:]))

    return num / denom
