from models.predictor import Predictor
import numpy as np
from scipy.special import expit
import misc.constants as cs
from sklearn.linear_model import Ridge


class ELM(Predictor):
    """
    The ELM predictor is based on extreme learning machine neural networks by Huang et al (2006).
    Parameters:
        - self.params["hist"], history length
        - self.params["neurons"], number of neurons in the single hidden layer
        - self.params["l2"], L2 penalty added to the weights during fitting
    """

    def fit(self):
        print("fit",self.params["neurons"])
        # get training data
        x, y = self._str2dataset("train")

        n_inputs = int(np.shape(x)[1])

        # initialize the random hidden layer with weights and bias
        np.random.seed(cs.seed)
        self.W = np.random.normal(size=(n_inputs, int(self.params["neurons"])))
        self.b = np.random.normal(size=(1, int(self.params["neurons"])))

        # compute the outputs of the hidden layer
        H = expit(x @ self.W + self.b)

        # fit the outputs of the hidden layer to the objective values
        self.model = Ridge(self.params["l2"])
        self.model.fit(H, y)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y = self._str2dataset(dataset)

        # compute the output of the hidden layer
        H = expit(x @ self.W + self.b)

        # predict
        y_pred = self.model.predict(H)
        y_true = y.values

        return y_true, y_pred

    # def _reshape(self, data):
    #     y = data["y"]
    #     x = data.drop("y", axis=1)
    #     return x, y
