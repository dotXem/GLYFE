from misc.utils import printd
import sys
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from misc.constants import day_len


class Predictor(ABC):
    def __init__(self, subject, ph, params, train, valid, test):
        self.subject = subject
        self.params = params
        self.ph = ph

        self.train_x, self.train_y = self._reshape(train)
        self.valid_x, self.valid_y = self._reshape(valid)
        self.test_x, self.test_y = self._reshape(test)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass

    def _reshape(self, data):
        """
        Reshape the training, validation and testing datasets according to the given history length and prediction horizon.
        The goal is to compute, for every day and every split, the x and y samples.
        len(X)=len(y)=1440; len(X[i]) = hist*3+1, len(y[i]) = 1.
        :param data: dataset (array of pandas DataFrame) that needs to be reshaped;
        :return: training, validation and testing sets that have been reshaped.
        """

        data_r = []
        for day in data:
            data_r.append(self.__reshape_day(day))
        data_df = pd.concat(data_r)

        y = data_df["y"]
        x = data_df.drop("y", axis=1)

        return x, y

    def _str2dataset(self, dataset_name):
        if dataset_name in ["train", "training"]:
            return self.train_x, self.train_y
        elif dataset_name in ["valid", "validation"]:
            return self.valid_x, self.valid_y
        elif dataset_name in ["test", "testing"]:
            return self.test_x, self.test_y
        else:
            printd("Dataset name not known.")
            sys.exit(-1)

    def __reshape_day(self, day):
        hist, ph = self.params["hist"], self.ph
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

        new_day = pd.DataFrame(new_data, columns=new_columns)

        return new_day
