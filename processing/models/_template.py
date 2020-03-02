from processing.models.predictor import Predictor


class TEMPLATE(Predictor):
    """
    The TEMPLATE model is a template model for the benchmark. It is equivalent to the BASE model.
    To create a new model, one should write the 'fit', 'predict' and '_reshape' functions.

    To access the training, validation and testing sets, use: self.train_x and self.train_y; self.valid_x, self.valid_y;
    self.test_x, self.test_y.
    """

    def fit(self):
        """ The training of the model happens here (with the training set. The function does not return anything. """
        pass

    def predict(self, dataset):
        """
        The postprocessing of the model happens here (usually either with the validation or tesing sets)
        :param dataset: takes the x and y samples from the appropriate set. Possible values: "train", "valid" or "test".
        :return: true glucose values, predicted glucose values
        """
        x, y = self._str2dataset(dataset)
        y_pred = x.values
        y_true = y.values
        return y_true, y_pred

    def _reshape(self, data):
        """
        By default, it computes x and y from data (dataframe), y being the objective (glucose at prediction horizon ph),
        and x being de inputs features (time of prediction, history of glucose, history of CHO and insulin).
        It can be overridden in the child class (but usually not needs to be).
        :param data: pandas dataframe;
        :return: input features dataframe (self.x), objective (self.y);
        """
        y = data["y"]
        glucose_index = self.params["hist"] - 1
        x = data["glucose_" + str(glucose_index)]
        return x, y
