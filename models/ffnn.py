from misc.utils import printd
import os
import re
from models.predictor import Predictor
import keras
from keras.layers import Dense, AlphaDropout
import numpy as np
import misc.constants as cs


class FFNN(Predictor):
    """
    The FFNN predictor is based on feed-forward neural network and SELU activation functions.
    Parameters:
        - self.params["hist"], history length
        - self.params["hidden_neurons"], number of neurons per hidden layer
        - self.params["dropout"], alpha dropout rate (should be very low, e.g., 0.1)
        - self.params["l1"], L1 penalty added to the weights
        - self.params["l2"], l2 penalty added to the weights
        - self.params["max_epochs"], maximum number of epochs during training
        - self.params["batch_size"], size of the mini-batches
        - self.params["learning_rate_init"], initial learning rate
        - self.params["learning_rate_decay"], decay factor of the learning rate
    """

    def fit(self):
        # get training data
        x_train, y_train = self._str2dataset("train")
        x_valid, y_valid = self._str2dataset("valid")

        # create the model
        self.model = self._create_model()

        # save model
        rnd = np.random.randint(1e7)
        self.filepath = os.path.join("misc", "tmp_weights", "weights." + str(rnd) + ".hdf5")
        printd("Saved model's file:", self.filepath)

        # create checkpoint that saves the best model
        checkpoint = keras.callbacks.ModelCheckpoint(self.filepath, monitor="val_loss", verbose=1, save_best_only=True,
                                                     mode="min")
        callbacks_list = [checkpoint]

        self.model.fit(x_train, y_train, epochs=self.params["max_epochs"], batch_size=self.params["batch_size"],
                       validation_data=(x_valid, y_valid),
                       callbacks=callbacks_list, verbose=2)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y = self._str2dataset(dataset)

        # create the model
        self.model = self._create_model(self.filepath)

        # predict
        y_pred = self.model.predict(x, batch_size=self.params["batch_size"])
        y_true = y.values

        return y_true, y_pred

    def _create_model(self, weights=None):
        np.random.seed(cs.seed)

        hidden = self.params["hidden_neurons"]
        l1, l2 = self.params["l1"], self.params["l2"]
        n_inputs = np.shape(self.train_x)[1]
        dropout = self.params["dropout"]
        lri, lru = self.params["learning_rate_init"], self.params["learning_rate_decay"]

        act = keras.activations.selu
        reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

        # create the sequential model
        model = keras.models.Sequential()
        model.add(Dense(hidden[0], input_shape=(n_inputs,), activation=act, kernel_regularizer=reg))
        for layer in hidden[1:]:
            model.add(AlphaDropout(dropout, seed=cs.seed))
            model.add(Dense(layer, activation=act, kernel_regularizer=reg))
        model.add(AlphaDropout(dropout, seed=cs.seed))
        model.add(Dense(1))

        # pretraining of the model based on the subject and ph
        if weights:
            model.load_weights(weights)
        else:
            model.load_weights(self._get_weights(self.subject, self.ph))

        model.compile(loss=keras.losses.mean_absolute_error,
                      optimizer=keras.optimizers.Adam(lr=lri, decay=lru))

        return model

    def _get_weights(self, subject, ph):
        exp = r"(.*)#.*"
        pop = re.match(exp, subject).group(1)

        file = "weights." + pop + "." + str(ph) + ".hdf5"
        path = os.path.join(cs.path, "misc", "ffnn_weights", file)

        return path
