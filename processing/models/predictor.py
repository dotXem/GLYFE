from misc.utils import printd
import sys
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Predictor(ABC):
    def __init__(self, subject, ph, params, train, valid, test):
        self.subject = subject #TODO remove useless
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

    def _reshape(self, data):
        """
        Extract (and reshape if needed, depending on the model) the time, inputs, outputs of the data samples
        :param data: pandas DataFrame containing the samples;
        :return: time of samples, inputs samples, outputs samples
        """

        t = data["datetime"]
        y = data["y"]
        # x = data.drop(["y","datetime"], axis=1)
        x = data.drop(["y","datetime","time"], axis=1)

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