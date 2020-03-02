import numpy as np
import misc.constants as cs
from processing.models.predictor import Predictor
from statsmodels.tsa.statespace.sarimax import SARIMAX

#TODO remove

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
        self.params = params.copy()
        self.ph = ph

        self.params["hist"] //= cs.freq
        self.params["p"] //= cs.freq
        self.params["q"] //= cs.freq

        self.train_endog, self.train_exog = self._reshape(train)
        self.valid_endog, self.valid_exog, self.valid_target, self.valid_t = self._reshape_test(valid)
        self.test_endog, self.test_exog, self.test_target, self.test_t = self._reshape_test(test)

        self.data_dict = {
            "train": [self.train_endog, self.train_exog],
            "valid": [self.valid_endog, self.valid_exog, self.valid_target, self.valid_t],
            "test": [self.test_endog, self.test_exog, self.test_target, self.test_t],
        }

    def fit(self):
        # get training data
        [endog, exog] = self.data_dict["train"]
        p, d, q = int(self.params["p"]), int(self.params["d"]), int(self.params["q"])

        start_params = None

        self.model = SARIMAX(endog=endog,
                             exog=exog,
                             order=(p, d, q),
                             # simple_differencing=True,
                             enforce_invertibility=False,
                             # enforce_stationarity=False,
                             ).fit(disp=0,
                                   method="powell",
                                   start_params=start_params,
                                   # maxiter=200
                                   )

    def predict(self, dataset):
        # get the data for which we make the predictions
        [endog, exog, y_true, t] = self.data_dict[dataset]
        p, d, q, use_exog = int(self.params["p"]), int(self.params["d"]), int(self.params["q"]), int(self.params["use_exog"])
        ph = self.ph

        if use_exog:
            oos_exog = np.min(np.min(exog, axis=0), axis=0)


        y_pred = []
        for endog_i, exog_i in zip(endog, exog):
            if use_exog:
                model = self.model.apply(endog_i, exog_i)
                oos_exog_i = np.array([oos_exog.copy() for _ in range(ph)])
                preds = model.forecast(steps=ph, exog=oos_exog_i)
            else:
                model = self.model.apply(endog_i, exog_i)
                preds = model.forecast(steps=ph)
            y_pred.append(preds[-1])

        # print(np.shape(y_pred), np.shape(y_true))
        return self._format_results(y_true, y_pred, t)

    def _reshape(self, data):
        """
        Totally rewrite the reshape function as the structure needed to fit and predict the models is not the same.
        In particular, we need to compute the endogenous vector and the exogenous vector instead of x and y.
        :param data: array of pandas dataframe containing the data
        """
        hist = self.params["hist"]
        use_exog = int(self.params["use_exog"])

        # create groups of continuous time-series (because of the crossèvalidation rearranging
        df = data.copy()
        df = df.sort_values(by="datetime")
        df = df.resample(str(cs.freq) + 'min', on="datetime").mean()

        min_cho = df.min(axis=0).loc[["CHO_" + str(i) for i in range(hist)]]
        min_ins = df.min(axis=0).loc[["insulin_" + str(i) for i in range(hist)]]

        endog = df.loc[:, "glucose_" + str(hist - 1)].values
        if use_exog:
            exog = df.loc[:, ["CHO_" + str(hist - 1), "insulin_" + str(hist - 1)]].values

            # fill exog nans with default values (standardized 0) as nans are not accepted by SARIMAX
            nan_idx = np.where(np.isnan(exog))[0]
            default_exog = np.reshape([min_cho[-1],min_ins[-1]],(1,-1))
            exog[nan_idx] = np.concatenate([default_exog for _ in range(len(nan_idx))],axis=0)
            pass
        else:
            exog = None

        return endog, exog

    def _reshape_test(self, data):
        hist = self.params["hist"]
        use_exog = int(self.params["use_exog"])
        p = int(self.params["p"])

        endog_cols = ["glucose_" + str(i) for i in range(hist - p, hist)]

        endog = data.loc[:, endog_cols].values

        if use_exog:
            cho = np.expand_dims(data.loc[:, ["CHO_" + str(i) for i in range(hist - p, hist)]].values, axis=-1)
            ins = np.expand_dims(data.loc[:, ["insulin_" + str(i) for i in range(hist - p, hist)]].values, axis=-1)
            exog = np.concatenate([cho, ins], axis=2)
        else:
            exog = np.full((len(endog)), None)

        target = data.loc[:, "y"].values

        t = data.loc[:, "datetime"]

        return endog, exog, target, t
