from misc.utils import printd
import numpy as np
import misc.constants as cs
from processing.models.predictor import Predictor
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARIMAX(Predictor):
    """
    The ARIMAX predictor is based on Gaussian Processes and DotProduct kenerl.
    Parameters:
        - self.params["hist"], history length
        - self.params["p"], endogenous order (in minutes)
        - self.params["d"], integration order
        - self.params["q"], moving average order (in minutes)
        - self.params["e"], exogenous order (in minutes) - control the history (CHO, insuline) as input
    """

    def __init__(self, subject, ph, params, train, valid, test):
        self.subject = subject
        self.params = params.copy()
        self.ph = ph

        self.params["hist"] = int(self.params["hist"] // cs.freq)
        self.params["p"] = int(self.params["p"] // cs.freq)
        self.params["q"] = int(self.params["q"] // cs.freq)

        self.train_endog, self.train_exog = self._reshape(train)
        self.valid_endog, self.valid_exog, self.valid_exog_oos, self.valid_target, self.valid_t = self._reshape_test(
            valid)
        self.test_endog, self.test_exog, self.test_exog_oos, self.test_target, self.test_t = self._reshape_test(test)

        self.data_dict = {
            "train": [self.train_endog, self.train_exog],
            "valid": [self.valid_endog, self.valid_exog, self.valid_exog_oos, self.valid_target, self.valid_t],
            "test": [self.test_endog, self.test_exog, self.test_exog_oos, self.test_target, self.test_t],
        }

    def fit(self):
        # get training data
        [endog, exog] = self.data_dict["train"]
        p, d, q = self.params["p"], self.params["d"], self.params["q"]

        self.model = SARIMAX(endog=endog,
                             exog=exog,
                             order=(p, d, q),
                             enforce_invertibility=False,
                             ).fit(disp=1,
                                   method="powell",
                                   )

    def predict(self, dataset):
        # get the data for which we make the predictions
        [endog, exog, exog_oos, y_true, t] = self.data_dict[dataset]
        ph = self.ph

        y_pred = []
        for endog_i, exog_i, exog_oos_i in zip(endog, exog, exog_oos):
            model = self.model.apply(endog_i, exog_i)
            preds = model.forecast(steps=ph, exog=exog_oos_i)
            y_pred.append(preds[-1])

        printd("end predict")

        return self._format_results(y_true, y_pred, t)

    def _reshape(self, data):
        """
        Totally rewrite the reshape function as the structure needed to fit and predict the models is not the same.
        In particular, we need to compute the endogenous vector and the exogenous vector instead of x and y.
        :param data: array of pandas dataframe containing the data
        """
        hist, p, use_exog = self.params["hist"], self.params["p"], self.params["use_exog"]

        # create groups of continuous time-series (because of the cross-validation rearranging
        df = data.copy()
        df = df.sort_values(by="datetime")
        df = df.resample(str(cs.freq) + 'min', on="datetime").mean()

        min_cho = df.min(axis=0).loc[["CHO_" + str(i) for i in range(hist)]][hist - p:].values.reshape(1, -1)
        min_ins = df.min(axis=0).loc[["insulin_" + str(i) for i in range(hist)]][hist - p:].values.reshape(1, -1)

        endog = df.loc[:, "glucose_" + str(hist - 1)].values
        if use_exog:
            cho = df.loc[:, ["CHO_" + str(i) for i in range(hist - p, hist)]].values
            ins = df.loc[:, ["insulin_" + str(i) for i in range(hist - p, hist)]].values

            nans_ind = np.unique(np.where(np.isnan(cho))[0])
            cho[nans_ind] = min_cho * np.ones((len(nans_ind), 1))
            ins[nans_ind] = min_ins * np.ones((len(nans_ind), 1))

            exog = np.c_[cho, ins]
        else:
            exog = None

        return endog, exog

    def _reshape_test(self, data):
        hist, p, use_exog = self.params["hist"], self.params["p"], int(self.params["use_exog"])
        ph = self.ph

        # create groups of continuous time-series (because of the cross-validation rearranging
        df = data.copy()
        df = df.sort_values(by="datetime")
        df = df.resample(str(cs.freq) + 'min', on="datetime").mean()

        endog_cols = ["glucose_" + str(i) for i in range(hist - p, hist)]

        t = df.index
        endog = df.loc[:, endog_cols].values

        if use_exog:

            def compute_exog_oos(exog_name):
                columns = [exog_name + "_" + str(i) for i in range(hist - p, hist)]
                min = df.min(axis=0).loc[columns].values.reshape(1, -1)
                exog = df.loc[:, columns].values
                exog[np.unique(np.where(np.isnan(exog))[0])] = min
                exog = np.concatenate([min.repeat(p - 1, axis=0), exog], axis=0)
                exog = np.transpose(np.flip([exog[(p - 1) - i:len(exog) - i] for i in range(p)], axis=0), (1, 0, 2))

                exog_oos = np.expand_dims(min.repeat(ph, axis=0), axis=0).repeat(len(exog), axis=0)

                return exog, exog_oos

            cho_exog, cho_oos = compute_exog_oos("CHO")
            ins_exog, ins_oos = compute_exog_oos("insulin")
            exog = np.concatenate([cho_exog, ins_exog], axis=2)
            exog_oos = np.concatenate([cho_oos, ins_oos], axis=2)
        else:
            exog = np.full((len(endog)), None)
            exog_oos = exog.copy()

        target = df.loc[:, "y"].values
        na_idx = np.where(~np.isnan(target))[0]

        return endog[na_idx], exog[na_idx], exog_oos[na_idx], target[na_idx], t[na_idx]
