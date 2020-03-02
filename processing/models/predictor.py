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

        self.train_x, self.train_y, self.train_t = self._reshape(train)
        self.valid_x, self.valid_y, self.valid_t = self._reshape(valid)
        self.test_x, self.test_y, self.test_t = self._reshape(test)

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, dataset):
        pass

    #TODO rework description
    def _reshape(self, data):
        """
        Reshape the training, validation and testing datasets according to the given history length and prediction horizon.
        The goal is to compute, for every day and every split, the x and y samples.
        len(X)=len(y)=1440; len(X[i]) = hist*3+1, len(y[i]) = 1.
        :param data: dataset (array of pandas DataFrame) that needs to be reshaped;
        :return: training, validation and testing sets that have been reshaped.
        """

        t = data["datetime"]
        y = data["y"]
        x = data.drop(["y","datetime"], axis=1)

        return x, y, t

    def _str2dataset(self, dataset_name):
        if dataset_name in ["train", "training"]:
            return self.train_x, self.train_y, self.train_t
        elif dataset_name in ["valid", "validation"]:
            return self.valid_x, self.valid_y, self.valid_t
        elif dataset_name in ["test", "testing"]:
            return self.test_x, self.test_y, self.test_t
        else:
            printd("Dataset name not known.")
            sys.exit(-1)

    def _format_results(self, y_true, y_pred, t):
        return pd.DataFrame(data=np.c_[y_true,y_pred],index=pd.DatetimeIndex(t.values),columns=["y_true", "y_pred"])