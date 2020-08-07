import numpy as np
from misc.utils import printd
import matplotlib.pyplot as plt


def remove_anomalies(data, anomalies_threshold=2.5, n_run=5, disp=False):
    """
    Remove glucose anomalies within the signals.
    :param data: time-series Dataframe
    :param anomalies_threshold: anomaly detection threshold
    :param n_run: number of times to run the algorithm
    :param disp: if the results of the removal shall be plotted and printed
    :return: Dataframe with no anomaly
    """
    data_no_anomaly = data.copy()
    for i in range(n_run):
        anomalies_indexes = detect_glucore_readings_anomalies(data_no_anomaly, threshold=anomalies_threshold)
        data_no_anomaly = data_no_anomaly.drop(anomalies_indexes, axis=0)
        data_no_anomaly = data_no_anomaly.reset_index(drop=True)
        if disp:
            printd("[iter {}] Number of anomalies removed : {}".format(i, len(anomalies_indexes)))

    if disp:
        plot(data, data_no_anomaly)

    return data_no_anomaly

def plot(data, data_no_anomaly):
    data.plot("datetime", "glucose")
    data_no_anomaly.plot("datetime", "glucose")
    plt.show()

def detect_glucore_readings_anomalies(data, threshold):
    """
    Detect contextual glucose anomalies characterized by having very high or low amplitude compared to
    the other surrounding values.
    :param data: dataframe
    :param threshold: z_score threshold flagging samples as anomalies. Samples should have the to and from variations h
    aving a z_score > threshold to be flagged as anomalies
    :return:
    """
    df_nonan = data.drop(["CHO", "insulin"], axis=1).dropna()
    i = df_nonan.index
    t = df_nonan["datetime"].astype(np.int64).values
    g = df_nonan["glucose"].values

    arr = np.concatenate([g.reshape(-1, 1), t.reshape(-1, 1)], axis=1)

    # compute the contexts
    contexts = [arr[i - 1:i + 2] for i in range(1, len(arr) - 2)]

    # compute the variations
    # variations = np.array([np.divide(np.diff(context[:, 1]), np.diff(context[:, 0])) for context in contexts])
    variations = np.diff(contexts, axis=1)
    variations = np.divide(variations[:, :, 0], variations[:, :, 1])

    # compute the behavior of the variations
    mean = np.nanmean(variations)
    std = np.nanstd(variations)

    # compute z_score
    z_score = np.divide(np.subtract(variations, mean), std)

    # flag variations that are anomalies
    anomalies_indexes = np.array(np.where(np.all(
        np.c_[np.abs(z_score)[:, 0] >= threshold, np.abs(z_score)[:, 1] >= threshold, np.prod(z_score, axis=1) < 0],
        axis=1))).reshape(-1, 1) + 1

    return i[anomalies_indexes].ravel()
