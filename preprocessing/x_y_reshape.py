from misc.constants import day_len
import numpy as np
import pandas as pd

def x_y_reshape(train, valid, test, hist, ph):
    """
    Reshape the training, validation and testing datasets according to the given history length and prediction horizon.
    The goal is to compute, for every day and every split, the x and y samples.
    len(X)=len(y)=1440; len(X[i]) = hist*3+1, len(y[i]) = 1.
    :param train: training dataset (array of pandas DataFrame);
    :param valid: validation dataset (array of pandas DataFrame);
    :param test: testing dataset (array of pandas DataFrame);
    :param hist: history length (in minutes)
    :param ph: prediction horizon (in minutes)
    :return: training, validation and testing sets that have been reshaped.
    """
    def reshape_day(day):
        data = day.values
        new_data = np.c_[data[hist:hist + day_len, 0].reshape(-1, 1),
                     np.array([data[i:i + day_len, 1] for i in range(hist)]).transpose(),
                     np.array([data[i:i + day_len, 2] for i in range(hist)]).transpose(),
                     np.array([data[i:i + day_len, 3] for i in range(hist)]).transpose(),
                     data[hist + ph:hist + ph + 1440, 1].reshape(-1, 1)]
        new_columns = np.r_[[day.columns[0]],
                            [day.columns[1] + "_" + str(i) for i in range(hist)],
                            [day.columns[2] + "_" + str(i) for i in range(hist)],
                            [day.columns[3] + "_" + str(i) for i in range(hist)],
                            ["y"]]

        new_day = pd.DataFrame(new_data,columns=new_columns)
        
        return new_day

    train_r, valid_r, test_r = [], [], []
    for split in range(len(train)):
        # train reshape
        new_train_tmp = []
        for day in train[split]:
            new_train_tmp.append(reshape_day(day))
        train_df = pd.concat(new_train_tmp)
        train_r.append(train_df)

        # valid reshape
        new_valid_tmp = []
        for day in valid[split]:
            new_valid_tmp.append(reshape_day(day))
        valid_df = pd.concat(new_valid_tmp)
        valid_r.append(valid_df)

        # test reshape
        new_test_tmp = []
        for day in test[split]:
            new_test_tmp.append(reshape_day(day))
        test_df = pd.concat(new_test_tmp)
        test_r.append(test_df)

    return train_r, valid_r, test_r
