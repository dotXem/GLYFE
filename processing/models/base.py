from processing.models.predictor import Predictor


class BASE(Predictor):
    """
    The BASE model predict at horizon PH (i.e., at time t+PH) that the glucose value is the same as at time t.
    """

    def fit(self):
        pass

    def predict(self, dataset):
        x, y, t = self._str2dataset(dataset)
        g_col = [col for col in x.columns if "glucose" in col]
        g_col = [col[8:] for col in g_col]
        last_glucose_idx = "glucose_" + str(max(g_col))
        y_pred = x.loc[:, last_glucose_idx].values
        y_true = y.values
        return self._format_results(y_true, y_pred, t)
