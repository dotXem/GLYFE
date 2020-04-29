from torch.utils.data import TensorDataset
import torch
from misc.utils import printd
import os
from processing.models.predictor import Predictor
import numpy as np
import torch.nn as nn
from .pytorch_tools.training import fit, predict
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
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")

        # save model
        rnd = np.random.randint(1e7)
        self.checkpoint_file = os.path.join(cs.path, "tmp", "ffnn_weights", str(rnd) + ".pt")
        printd("Saved model's file:", self.checkpoint_file)

        self.model = self.FFNN_Module(x_train.shape[1],self.params["hidden"], self.params["cell_type"], self.params["dropout"])
        self.model.cuda()
        self.loss_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

        train_ds = self.to_dataset(x_train, y_train)
        valid_ds = self.to_dataset(x_valid, y_valid)

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds, valid_ds, self.params["patience"], self.checkpoint_file)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self.to_dataset(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        # predict
        y_true, y_pred = predict(self.model, ds)

        return self._format_results(y_true, y_pred, t)

    class FFNN_Module(nn.Module):
        def __init__(self, n_in, neurons, cell, dropout):
            super().__init__()

            if cell == nn.SELU:
                dropout_func = nn.AlphaDropout
            else:
                dropout_func = nn.Dropout

            n_in_vec = np.r_[n_in, neurons[:-1]]
            n_out_vec = neurons

            self.layers = nn.Sequential(
                *np.concatenate([self._create_layer(n_in, n_out, cell, dropout_func, dropout)
                                 for n_in, n_out in zip(n_in_vec, n_out_vec)], axis=0),
                nn.Linear(neurons[-1], 1)
            )

        def _create_layer(self, n_in, n_out, cell_type, dropout_func, dropout_rate):
            return [nn.Linear(n_in, n_out), cell_type(), dropout_func(dropout_rate)]

        def forward(self, xb):
            return self.layers(xb).reshape(-1)


    def to_dataset(self, x, y):
        return TensorDataset(torch.Tensor(x.values).cuda(), torch.Tensor(y.values).cuda())

