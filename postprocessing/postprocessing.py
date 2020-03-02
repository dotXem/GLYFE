from misc.constants import day_len
import numpy as np
import misc


def postprocessing(results, scalers, dataset):
    """
    Do the whole post-processing pipeline by:
    1. rescaling the results
    2. formatting the results
    :param results:
    :param scalers:
    :return:
    """
    freq = misc.datasets.datasets[dataset]["freq"]
    results = _rescale(results, scalers)
    results = _reshape(results, freq)

    return results


def _rescale(results, scalers):
    """
    Before evaluating the results we need to rescale the glucose predictions that have been standardized.
    :param results: array of shape (cv_fold, n_predictions, 2);
    :param scalers: array of scalers used to standardize the data;
    :return: rescaled results;
    """
    scaled_results = []
    for res, scaler in zip(results, scalers):
        mean = scaler.mean_[-1]
        std = scaler.scale_[-1]
        scaled_results.append(res * std + mean)

    return scaled_results


def _reshape(results, freq):
    """
    Reshape (resample) the results into the given sampling frequency
    :param results: array of dataframes with predictions and ground truths
    :param freq: sampling frequency
    :return: reshaped results
    """
    return [res_.resample(str(freq) + "min").mean() for res_ in results]
