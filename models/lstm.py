import pandas as pd
from misc.utils import printd
import os
import re
from models.predictor import Predictor
import keras
from keras.layers import Dense, AlphaDropout
import numpy as np
import misc.constants as cs


class LSTM(Predictor):
    """
    The LSTM predictor is based on long short-term memory recurrent neural network.
    Parameters:
        - self.params["hist"], history length
        - self.params["hidden_neurons"], number of neurons per hidden layer
        - self.params["dropout"], dropout rate
        - self.params["recurrent_dropout"], recurrent dropout rate
        - self.params["max_epochs"], maximum number of epochs during training
        - self.params["batch_size"], size of the mini-batches
        - self.params["learning_rate_init"], initial learning rate
        - self.params["learning_rate_decay"], decay factor of the learning rate
        - self.params["l1"], L1 penalty added to the weights
        - self.params["l2"], l2 penalty added to the weights
    """

    def fit(self):
        # get training data
        x_train, y_train = self._str2dataset("train")
        x_valid, y_valid = self._str2dataset("valid")

        # create the model
        self.model = self._create_model()

        # save model - the random number almost assures that the file does not exist in the directory
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

    def _reshape(self, data):
        """ because of the recurrent neural network architecture, we need to reshape the data"""

        hist, ph = self.params["hist"], self.ph

        new_columns = np.r_[[data[0].columns[0] + "_" + str(i) for i in range(hist)],
                            [data[0].columns[1] + "_" + str(i) for i in range(hist)],
                            [data[0].columns[2] + "_" + str(i) for i in range(hist)],
                            [data[0].columns[3] + "_" + str(i) for i in range(hist)],
                            ["y"]]

        data_r = []
        for day in data:
            val = day.values

            new_data = np.c_[np.array([val[i:i + cs.day_len, 0] for i in range(hist)]).transpose(),
                             np.array([val[i:i + cs.day_len, 1] for i in range(hist)]).transpose(),
                             np.array([val[i:i + cs.day_len, 2] for i in range(hist)]).transpose(),
                             np.array([val[i:i + cs.day_len, 3] for i in range(hist)]).transpose(),
                             val[hist + ph:hist + ph + 1440, 1].reshape(-1, 1)]

            data_r.append(pd.DataFrame(data=new_data, columns=new_columns))
        data_df = pd.concat(data_r)

        y = data_df["y"]
        x = data_df.drop("y", axis=1)

        x = x.values.reshape(-1, 4, self.params["hist"])
        x = np.rollaxis(x, 2, 1)
        #
        y = y.values.reshape(-1, 1)

        return x, y

    def _create_model(self, weights=None):
        np.random.seed(cs.seed)

        hidden = self.params["hidden_neurons"]
        l1, l2 = self.params["l1"], self.params["l2"]
        n_inputs = np.shape(self.train_x)[2]
        dropout = self.params["dropout"]
        recdrop = self.params["recurrent_dropout"]
        lri, lru = self.params["learning_rate_init"], self.params["learning_rate_decay"]
        unrolling = self.params["hist"]

        reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

        # create the sequential model
        model = keras.models.Sequential()
        seq = True if len(hidden) > 1 else False
        model.add(keras.layers.LSTM(hidden[0], input_shape=(unrolling, n_inputs), recurrent_dropout=recdrop,
                                    kernel_regularizer=reg, return_sequences=seq))
        for n in hidden[1:]:
            model.add(keras.layers.LSTM(n, recurrent_dropout=recdrop, dropout=dropout, kernel_regularizer=reg))
        model.add(keras.layers.Dense(1))

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
        path = os.path.join(cs.path, "misc", "lstm_weights", file)

        return path
