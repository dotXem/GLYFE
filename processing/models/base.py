import misc.constants as cs
from processing.models.predictor import Predictor


class BASE(Predictor):
    """
    The BASE model predict at horizon PH (i.e., at time t+PH) that the glucose value is the same as at time t.

    WARNING : due to the scaling to 0 mean and 1 variance of the input variables and the rescaling based on the output
    original mean and variance, the predictions at t+PH are not exactly the values at time t.
    To achieve such a result, one should comment out the scaling lines in the preprocessing.py and the
    postprocessing.py files.
    """

    def fit(self):
        pass

    def predict(self, dataset):
        x, y, t = self._str2dataset(dataset)
        hist_f = self.params["hist"] // cs.freq
        y_pred = x.loc[:,"glucose_" + str(hist_f-1)].values
        y_true = y.values
        return self._format_results(y_true, y_pred, t)
