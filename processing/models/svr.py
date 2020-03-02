from processing.models.predictor import Predictor
from sklearn.svm import SVR as skSVR


class SVR(Predictor):
    """
    The SVR predictor is based on Support Vector Regression.
    Parameters:
        - self.params["hist"], history length
        - self.params["kernel"], kernel to be used
        - self.params["C"], loss
        - self.params["epsilon"], wideness of the no-penalty tube
        - self.params["gamma"], kernel coefficient
        - self.params["shrinking"], wether or not to use the shrinkin heuristic
    """

    def fit(self):
        # get training data
        x, y, t = self._str2dataset("train")

        # define the model
        self.model = skSVR(
            C=self.params["C"],
            epsilon=self.params["epsilon"],
            kernel=self.params["kernel"],
            gamma=self.params["gamma"],
            # cache_size=self.CACHE_SIZE
            shrinking=True
        )

        # fit the model
        self.model.fit(x, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)

        # predict
        y_pred = self.model.predict(x)
        y_true = y.values

        return self._format_results(y_true, y_pred, t)
