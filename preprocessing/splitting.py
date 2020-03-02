import numpy as np
import pandas as pd
import sklearn.model_selection as sk_model_selection
import misc.constants as cs

def split(data, day_len, test_n_days, cv_factor):
    """
    Split samples into training, validation, and testing days. Testing days are the last test_n_days days, and training
    and validation sets are created by permuting of splits according to the cv_factor.
    :param data: dataframe of samples
    :param day_len: length of day in freq minutes
    :param test_n_days: number of testing days
    :param cv_factor: cross validation factor
    :return: training, validation, testing samples folds
    """
    # the test split is equal to the last test_n_days days of data
    test = [data.iloc[-test_n_days * day_len:].copy()]

    # train+valid samples = all minus first and test days
    fday_n_samples = data.shape[0] - (data.shape[0] // day_len * day_len)
    train_valid = data.iloc[fday_n_samples:-test_n_days * day_len].copy()

    # split train_valid into cv_factor folds for inner cross-validation
    n_days = train_valid.shape[0] // day_len
    days = np.arange(n_days)

    kf = sk_model_selection.KFold(n_splits=cv_factor, shuffle=True, random_state=cs.seed)

    train, valid = [], []
    for train_idx, valid_idx in kf.split(days):
        def get_whole_day(data, i):
            return data[i * day_len:(i + 1) * day_len]

        train.append(pd.concat([get_whole_day(train_valid, i) for i in train_idx], axis=0, ignore_index=True))
        valid.append(pd.concat([get_whole_day(train_valid, i) for i in valid_idx], axis=0, ignore_index=True))

    return train, valid, test