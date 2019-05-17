from sklearn.preprocessing import StandardScaler
import pandas as pd


def standardization(train, valid, test):
    """
    Standardize (zero-mean and unit-variance) the datasets.
    :param train: array of DataFrames representing the processing sets
    :param valid: array of DataFrames representing the validation sets
    :param test:  array of DataFrames representing the testing sets
    :return: the datasets standardized based on the processing sets
    """
    train_scaled, valid_scaled, test_scaled, scalers = [], [], [], []
    for train_set, valid_set, test_set in zip(train, valid, test):
        df_len = len(train_set[0].index)

        # merge the dataframes before standardizing
        train_set = pd.concat(train_set).reindex()
        valid_set = pd.concat(valid_set).reindex()
        test_set = pd.concat(test_set).reindex()

        # standardize
        scaler = StandardScaler()
        train_scaled_tmp = scaler.fit_transform(train_set)
        valid_scaled_tmp = scaler.transform(valid_set)
        test_scaled_tmp = scaler.transform(test_set)

        # reshape df = day
        train_scaled_tmp = [pd.DataFrame(train_scaled_tmp[i * df_len:(i + 1) * df_len], columns=["time","glucose","CHO","insulin"]) for i in
                            range(len(train_scaled_tmp) // df_len)]
        valid_scaled_tmp = [pd.DataFrame(valid_scaled_tmp[i * df_len:(i + 1) * df_len], columns=["time","glucose","CHO","insulin"]) for i in
                            range(len(valid_scaled_tmp) // df_len)]
        test_scaled_tmp = [pd.DataFrame(test_scaled_tmp[i * df_len:(i + 1) * df_len], columns=["time","glucose","CHO","insulin"]) for i in
                           range(len(test_scaled_tmp) // df_len)]

        # save the results
        train_scaled.append(train_scaled_tmp)
        valid_scaled.append(valid_scaled_tmp)
        test_scaled.append(test_scaled_tmp)
        scalers.append(scaler)

    return train_scaled, valid_scaled, test_scaled, scalers
