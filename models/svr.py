from models.predictor import Predictor
from sklearn.svm import SVR as skSVR
import misc.constants as cs


class SVR(Predictor):
    """
    The SVR predictor is based on Support Vector Regression.
    Parameters:
        - self.params["hist"], history length
        - self.params["kernel"], kernel to be used
        - self.params["C"], loss
        - self.params["epsilon"], wideness of the no-penalty tube
        - self.params["gamma"], kernel coefficient
        - self.params["shrinking"], wether or not tp ise the shrinkin heuristic
    """
    CACHE_SIZE = 8000  # in Mb, allow more cache to the fitting of the model; might need to be changed depending on the system.

    def fit(self):
        # get training data
        x, y = self._str2dataset("train")

        # define the model
        self.model = skSVR(C=self.params["C"],
                         epsilon=self.params["epsilon"],
                         kernel=self.params["kernel"],
                         gamma=self.params["gamma"],
                         shrinking=self.params["shrinking"],
                         cache_size=self.CACHE_SIZE)

        # fit the model
        self.model.fit(x, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y = self._str2dataset(dataset)

        # predict
        y_pred = self.model.predict(x)
        y_true = y.values

        return y_true, y_pred

    # def _reshape(self, data):
    #     y = data["y"]
    #     x = data.drop("y", axis=1)
    #     return x, y
