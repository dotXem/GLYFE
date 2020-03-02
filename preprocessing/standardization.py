import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize(train, valid, test):
    """
    Standardize (zero mean and unit variance) the sets w.r.t to the training set for every fold
    :param train: training sample fold
    :param valid: validation sample fold
    :param test: testing sample fold
    :return: standardized training, validation, and testing sets
    """
    columns = train[0].columns.drop("datetime")
    train_scaled, valid_scaled, test_scaled, scalers = [], [], [], []
    for i in range(len(train)):
        scaler = StandardScaler()

        # standardize the sets (-> ndarray) without datetime
        train_i = scaler.fit_transform(train[i].drop("datetime", axis=1))
        valid_i = scaler.transform(valid[i].drop("datetime", axis=1))
        test_i = scaler.transform(test[0].copy().drop("datetime", axis=1))

        # recreate dataframe
        train_i = pd.DataFrame(data=train_i, columns=columns)
        valid_i = pd.DataFrame(data=valid_i, columns=columns)
        test_i = pd.DataFrame(data=test_i, columns=columns)

        # add datetime
        train_i["datetime"] = pd.DatetimeIndex(train[i].loc[:, "datetime"].values)
        valid_i["datetime"] = pd.DatetimeIndex(valid[i].loc[:, "datetime"].values)
        test_i["datetime"] = pd.DatetimeIndex(test[0].loc[:, "datetime"].values)

        # reorder
        train_i = train_i.loc[:, train[i].columns]
        valid_i = valid_i.loc[:, valid[i].columns]
        test_i = test_i.loc[:, test[0].columns]

        # save
        train_scaled.append(train_i)
        valid_scaled.append(valid_i)
        test_scaled.append(test_i)

        scalers.append(scaler)

    return train_scaled, valid_scaled, test_scaled, scalers