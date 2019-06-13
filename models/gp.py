from models.predictor import Predictor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor
import misc.constants as cs


class GP(Predictor):
    """
    The GP predictor is based on Gaussian Processes and DotProduct kenerl.
    Parameters:
        - self.params["hist"], history length
        - self.params["alpha"], noise
        - self.params["sigma_0"], inhomogeneity coefficient
    """

    def fit(self):
        # get training data
        x, y = self._str2dataset("train")

        # define the GP kernel
        kernel = DotProduct(sigma_0=self.params["sigma_0"])

        # define the model
        self.model = GaussianProcessRegressor(kernel=kernel,
                                              alpha=self.params["alpha"],
                                              random_state=cs.seed)
        # fit the model
        self.model.fit(x, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y = self._str2dataset(dataset)

        # predict
        y_pred = self.model.predict(x)
        y_true = y.values

        return y_true, y_pred
