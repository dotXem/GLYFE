import sklearn.model_selection as sk_model_selection
import misc.constants as cs
import numpy as np


def train_valid_test_split(data, cv):
    """
    Compute processing, validation and testing sets following a given distribution.
    :param data: array of days of data (see formatting.raw_to_day_long);
    :param cv: cross-validation factor implying the distribution (cv-2)/cv; 1/cv; 1/cv %
    :return: training_sets, validation_sets, testing_sets
    """

    train, valid, test = [], [], []

    kf_1 = sk_model_selection.KFold(n_splits=cv, shuffle=True, random_state=cs.seed)
    for train_valid_index, test_index in kf_1.split(np.arange(len(data))):
        train_valid_fold = [data[i] for i in train_valid_index]
        test_fold = [data[i] for i in test_index]
        kf_2 = sk_model_selection.KFold(n_splits=cv - 1, shuffle=False)  # we already shuffled once
        for train_index, valid_index in kf_2.split(train_valid_fold):
            train_fold = [train_valid_fold[i] for i in train_index]
            valid_fold = [train_valid_fold[i] for i in valid_index]
            train.append(train_fold)
            valid.append(valid_fold)
            test.append(test_fold)

    return train, valid, test
