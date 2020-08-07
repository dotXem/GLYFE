from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


def analyze_filter(data, cutoff, order):
    """
    Plot the effect of a given low-pass filter on the data
    :param data: data
    :param cutoff: cutoff frequency
    :param order: order of filter
    :return: /
    """
    T = 5 * 60
    y = data.y.interpolate(method="linear").values
    N = len(y)
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    b, a = butter(order, cutoff, btype='lowpass', analog=False)
    y_filt = filtfilt(b, a, y)
    yf_filt = scipy.fftpack.fft(y_filt)

    plt.subplot(211)
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.plot(xf, 2.0 / N * np.abs(yf_filt[:N // 2]))

    plt.subplot(212)
    plt.plot(y)
    plt.plot(y_filt)

    plt.show()

def filter_splits_in_sample(data, day_len, mode, cutoff=0.333, order=1, plot=False):
    """
    Apply a given filter on the glucose history  of every sample for every fold dataframe
    :param data: sample dataframe
    :param day_len: length of day, scaled by sampling frequency
    :param mode: if "train", filter with future values
    :param cutoff: cutoff frequency of low-pass filter
    :param order: order of filter
    :param plot: if results should be plotted
    :return: filtered folds
    """
    data_hist_filtered = []
    for data_split in data:
        if plot:
            analyze_filter(data_split, cutoff, order)
        data_hist_filtered.append(filter_glucose_history(data_split, cutoff=cutoff, order=order))

    if mode == "train":
        data_horizon_filtered = []
        for data_split in data_hist_filtered:
            data_horizon_filtered.append(filter_glucose_horizon(data_split, day_len, cutoff=cutoff, order=order))

        return data_horizon_filtered
    else:
        return data_hist_filtered

def filter_glucose_history(data, cutoff, order):
    """
    Apply a given filter on the glucose history  of every sample for every fold dataframe
    :param data: sample dataframe
    :param cutoff: cutoff frequency of low-pass filter
    :param order: order of filter
    :return: filtered dataframe
    :return:
    """
    g_cols = [col for col in data.columns if "glucose" in col]
    g = data.loc[:, g_cols].copy(deep=True)

    new_cols = []
    # augment the necessary sample number for filtering, depends on the butterworth filter order
    padding = 3 * (order + 1) - len(g_cols) + 1
    for i in range(padding):
        col = "minus" + str(i + 1)
        new_cols.append(col)
        tmp_g = g.glucose_0.iloc[:-1 - i]
        tmp_g.index = g.glucose_0.index[i + 1:]
        g[col] = tmp_g
    g = g.loc[:, np.r_[np.flip(new_cols), g_cols]]

    b, a = butter(order, cutoff, btype='lowpass', analog=False)
    g_filtered = filtfilt(b, a, g.values, padlen=0)

    # first values could not be filtered because of nan, so we keep the unfiltered values
    g_filtered[:padding] = g.iloc[:padding].values

    for i, col in enumerate(g_cols):
        data.loc[:,col] = g_filtered[:,padding+i]

    return data


def filter_glucose_horizon(data, day_len, cutoff, order, test_n_days):
    """
    Apply a filter on the ground truth
    :param data: dataframe
    :param day_len: length of day, scaled by sampling frequency
    :param cutoff: cutoff frequency of low pass filter
    :param order: order of filter
    :param test_n_days: number of testing days (that can't be filtered)
    :return: filtered dataframe
    """
    train_y = data.y.iloc[:-test_n_days * day_len].values.reshape(-1, 1)

    # fill nans because we cant filter with them
    train_y_inter = data.y.iloc[:-test_n_days * day_len].interpolate(method="linear").values.reshape(-1, 1)
    first_notna = train_y_inter[~np.isnan(train_y_inter)][0]
    train_y_inter[np.isnan(train_y_inter)] = first_notna

    test_y = data.y.iloc[-test_n_days * day_len:].values.reshape(-1, 1)

    # filter
    b, a = butter(order, cutoff, btype='lowpass', analog=False)
    train_y_filt = filtfilt(b, a, train_y_inter.ravel()).reshape(-1, 1)

    # replace nans inside y
    train_y_filt[np.where(np.isnan(train_y))] = np.nan
    data.y = np.r_[train_y_filt, test_y]

    return data