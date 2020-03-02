import pandas as pd
import numpy as np

# Short-term prediction of glucose in type 1 diabetes using kernel adaptive filters - Georga

def time_gain(results, ph, freq, method="correlation"):
    """
    :param results: dataframe with predictions and ground truths
    :param ph: prediction horizon in minutes
    :param freq: sampling frequency in minutes
    :param method: how time-lag is computed either "correlation" or "mse"
    :return: the time anticipated by doing the predictions as PH-TL
    """
    return ph - time_lag(results, ph, freq, method=method)

def time_lag(results, ph, freq, method="correlation"):
    """
    Compute the time-lag (TL) metric, as the shifting number maximizing the correlation (or minimizing the MSE)
    between the prediction and the ground truth.
    :param results: dataframe with predictions and ground truths
    :return: mean of daily time-lags
    """
    df_resampled = results.resample(str(freq)  + "min").mean() # if not resampled already

    # arr = [df_resampled.corr().iloc[0, 1]]
    arr = []
    for i in range(ph // freq + 1
                   ):
        if i == 0:
            df_shifted = df_resampled.copy()
        else:
            df_shifted = pd.DataFrame(data=np.c_[df_resampled.values[:-i, 0], df_resampled.values[i:, 1]])

        if method == "correlation":
            arr.append(df_shifted.corr().iloc[0, 1])
        elif method == "mse":
            arr.append(df_shifted.diff(axis=1).pow(2).mean(axis=0).values[1])

    if method == "correlation":
        tl = np.argmax(arr) * freq
    elif method == "mse":
        tl = np.argmin(arr) * freq

    return tl

