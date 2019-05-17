import numpy as np

from models.predictor import Predictor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from misc.utils import printd


class ARIMAX(Predictor):
    """
    The GP predictor is based on Gaussian Processes and DotProduct kenerl.
    Parameters:
        - self.params["hist"], history length
        - self.params["p"], endogenous order (in minutes)
        - self.params["d"], integration order
        - self.params["q"], moving average order (in minutes)
        - self.params["e"], exogenous order (in minutes) - control the history (CHO, insuline) as input
    """

    def __init__(self, subject, ph, params, train, valid, test):
        self.subject = subject
        self.params = params
        self.ph = ph

        self.train_endog, self.train_exog = self._reshape(train)
        self.valid_endog, self.valid_exog = self._reshape(valid)
        self.test_endog, self.test_exog = self._reshape(test)

        self.data_dict = {
            "train": [self.train_endog, self.train_exog],
            "valid": [self.valid_endog, self.valid_exog],
            "test": [self.test_endog, self.test_exog],
        }

    def fit(self):
        # get training data
        [endog, exog] = self.data_dict["train"]
        p, d, q = int(self.params["p"]), int(self.params["d"]), int(self.params["q"])

        # for every day, fit the model and use the fitted parameters as input to the other one
        start_params = None
        for endog_d, exog_d in zip(endog, exog):
            self.model = SARIMAX(endog=endog_d,
                                 exog=exog_d,
                                 order=(p, d, q),
                                 enforce_stationarity=False,
                                 enforce_invertibility=False).fit(method="powell",
                                                                  start_params=start_params,
                                                                  maxiter=1000000,
                                                                  disp=0)
            start_params = self.model.params

    def predict(self, dataset):
        # get the data for which we make the predictions
        [endog, exog] = self.data_dict[dataset]
        p, d, q = int(self.params["p"]), int(self.params["d"]), int(self.params["q"])
        hist = self.params["hist"]
        ph = self.ph
        start_params = self.model.params

        y_pred, y_true = [], []
        for endog_d, exog_d in zip(endog, exog):
            model = SARIMAX(endog=endog_d,
                            exog=exog_d,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(method="powell",
                                                             start_params=start_params,
                                                             maxiter=1000000,
                                                             disp=0)
            for i in range(len(endog_d) - ph):
                # the first prediction of every day do not use all the endog samples, but rather the first one available
                if i < hist:
                    start = 0
                    end = ph + i
                    dynamic = i
                else:
                    start = i - hist
                    end = start + hist + ph
                    dynamic = hist

                y_pred.append(model.predict(start, end, dynamic)[-1])
                y_true.append(endog_d[end])

        return np.reshape(y_true,(-1,1)), np.reshape(y_pred,(-1,1))

    def _reshape(self, data):
        """
        Totally rewrite the reshape function as the structure needed to fit and predict the models is not the same.
        In particular, we need to compute the endogenous vector and the exogenous vector instead of x and y.
        :param data: array of pandas dataframe containing the data
        """
        hist = self.params["hist"]
        e = int(self.params["e"])
        n_samples = len(data[0])

        exog_indexes = np.arange(e) * np.ones((n_samples - hist, e)) + (hist - e) + np.arange(
            n_samples - hist).reshape(-1, 1)
        exog_indexes = exog_indexes.astype(int)

        endog, exog = [], []

        for day in data:
            endog_d = day.loc[hist:, "glucose"].values
            exog_d = np.c_[day.loc[hist:, "time"].values.reshape(-1, 1),
                           day.loc[:, ["CHO", "insulin"]].values[exog_indexes].reshape(-1, 2 * e)] \
                if exog_indexes.size != 0 else None

            endog.append(endog_d)
            exog.append(exog_d)

        return endog, exog
