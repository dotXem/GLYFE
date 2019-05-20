from models.predictor import Predictor


class BASE(Predictor):
    """
    The BASE model predict at horizon PH (i.e., at time t+PH) that the glucose value is the same as at time t.
    """

    def fit(self):
        # There is no fitting here.
        pass

    def predict(self, dataset):
        x, y = self._str2dataset(dataset)
        y_pred = x.values
        y_true = y.values
        return y_true, y_pred

    def _reshape(self, data):
        """
        Override the reshape function as we only needs one input feature: the glucose value at the time of the
        prediction (i.e., the last known value of glucose)
        """
        x, y = super()._reshape(data)
        glucose_index = self.params["hist"] - 1
        x = x.loc[:, "glucose_" + str(glucose_index)]
        return x, y
