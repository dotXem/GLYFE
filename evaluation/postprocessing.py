def rescale(results, scalers):
    """
    Before evaluating the results we need to rescale the glucose predictions that have been standardized.
    :param results: array of shape (cv_fold, n_predictions, 2);
    :param scalers: array of scalers used to standardize the data;
    :return: rescaled results;
    """
    for i in range(len(scalers)):
        mean = scalers[i].mean_[1]
        std = scalers[i].scale_[1]
        results[:,i,:] = results[:,i,:] * std + mean
    return results
