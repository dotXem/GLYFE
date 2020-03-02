from processing.models.predictor import Predictor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class POLY(Predictor):
    """
    The POLY predictor is based on polynomial regression using only the time of the prediction to make the prediction.
    Parameters:
        - self.params["hist"], history length
        - self.params["degree"], degree of the polynomial features used to fit the model
    """

    def fit(self):
        # get training data
        x, y, t = self._str2dataset("train")
        x = x["time"].values.reshape(-1, 1)

        # compute the polynomial features
        self.poly_transform = PolynomialFeatures(degree=int(self.params["degree"]))
        x = self.poly_transform.fit_transform(x)

        # define the model
        self.model = LinearRegression()

        # fit the model
        self.model.fit(x, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        x = x["time"].values.reshape(-1, 1)

        # compute the polynomial features
        x = self.poly_transform.transform(x)

        # predict
        y_pred = self.model.predict(x)
        y_true = y.values

        return self._format_results(y_true, y_pred, t)
