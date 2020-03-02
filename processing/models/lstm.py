from .pytorch_tools.lstm_variants import LSTM_Gal
from torch.utils.data import TensorDataset
import torch
from misc.utils import printd
import os
import re
from processing.models.predictor import Predictor
import numpy as np
import misc.constants as cs
import torch.nn as nn
from .pytorch_tools.training import fit, predict

#TODO clean after results

class LSTM(Predictor):
    def fit(self):
        # get training data
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")

        # save model
        rnd = np.random.randint(1e7)
        self.checkpoint_file = os.path.join(cs.path, "tmp", "lstm_weights", str(rnd) + ".pt")
        printd("Saved model's file:", self.checkpoint_file)

        self.model = self.LSTM_Module(x_train.shape[2], self.params["hidden"], self.params["dropi"],
                                      self.params["dropw"], self.params["dropo"])
        self.model.cuda()
        self.loss_func = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

        train_ds = self.to_dataset(x_train, y_train)
        valid_ds = self.to_dataset(x_valid, y_valid)

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds, valid_ds,
            self.params["patience"], self.checkpoint_file)

    def predict(self, dataset):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self.to_dataset(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        # predict
        # y_pred = self.model.predict(x, batch_size=self.params["batch_size"])
        y_true, y_pred = predict(self.model, ds)

        return self._format_results(y_true, y_pred, t)

    def _reshape(self, data):
        time_step = data.time.diff().max()

        # extract data from data df
        t = data["datetime"]
        y = data["y"]

        g = data.loc[:, [col for col in data.columns if "glucose" in col]].values
        cho = data.loc[:, [col for col in data.columns if "CHO" in col]].values
        ins = data.loc[:, [col for col in data.columns if "insulin" in col]].values

        time_step_arr = np.flip(np.arange(g.shape[1]) * time_step).reshape(1, -1)
        t_x = data["time"].values.reshape(-1, 1) * np.ones((1, np.shape(time_step_arr)[1])) - time_step_arr

        # reshape timeseties in (n_samples, hist, 1) shape and concatenate
        g = g.reshape(-1, g.shape[1], 1)
        cho = cho.reshape(-1, g.shape[1], 1)
        ins = ins.reshape(-1, g.shape[1], 1)
        # t_x = t_x.reshape(-1, g.shape[1], 1)

        # x = np.concatenate([g, cho, ins, t_x], axis=2)
        x = np.concatenate([g, cho, ins], axis=2)

        return x, y, t

    class LSTM_Module(nn.Module):

        # def __init__(self, n_in, embedding_size, neurons, embedding_dropout, layer_dropout, recurrent_dropout):
        # def __init__(self, n_in, neurons, layer_dropout, recurrent_dropout):
        def __init__(self, n_in, neurons, dropi, dropw, dropo):
            super().__init__()
            # TODO handle embedding dropout and layer dropout for LSTM_Gal

            # if embedding_size is not None:
            #     self.emb_linear = nn.Linear(n_in, embedding_size, bias=False)
            #     self.emb_drop = nn.Dropout(embedding_dropout)
            # else:
            #     self.emb_linear = None
            #     embedding_size = n_in

            # self.lstm = BETTER_LSTM(n_in, neurons[0], len(neurons), dropouti=dropi, dropoutw=dropw, dropouto=dropo, batch_first=True)

            if not dropw == 0.0:
                # self.lstm = [LSTM_Gal(embedding_size, neurons[0], recurrent_dropout, batch_first=True)]
                self.lstm = [LSTM_Gal(n_in, neurons[0], dropw, batch_first=True).cuda()]
                self.dropouts = []
                for i in range(len(neurons[1:])):
                    self.dropouts.append(nn.Dropout(dropo))
                    self.lstm.append(LSTM_Gal(neurons[i], neurons[i + 1], dropw, batch_first=True).cuda())
                    # if (not i == len(neurons[1:]) - 1):
                self.dropouts.append(nn.Dropout(0.0))
                self.lstm = nn.Sequential(*self.lstm)
                self.dropouts = nn.Sequential(*self.dropouts)
            else:
                # self.lstm = nn.LSTM(embedding_size, neurons[0], len(neurons), dropout=layer_dropout, batch_first=True)
                self.lstm = nn.LSTM(n_in, neurons[0], len(neurons), dropout=dropo, batch_first=True)

            self.linear = nn.Linear(neurons[-1], 1)

            print(self.lstm)
            # r = 3
            # q = 512
            # p = q
            # m = 3
            # dp = 0.95

            # batch_first=True => input/ouput w/ shape (batch,seq,feature)
            # self.lstm = cs.LSTM(r,q)
            # self.lstm = LSTM_Gani(r, q, dp, True)
            # self.lstm = LSTM_Semenuita(r, q, dp, True)
            # self.lstm = nn.LSTM(r,q,1, batch_first=True)

            # self.linear = nn.Linear(q, 1)
            # self.dropout = nn.Dropout(0.25)

        def forward(self, xb):
            # xb, _ = self.lstm(xb)
            # xb = self.dropout(xb[:, -1, :])
            # xb = self.linear(xb)

            # if self.emb_linear is not None:
            #     xb = self.emb_drop(self.emb_linear(xb))

            if self.lstm.__class__.__name__ == "LSTM":
                xb, _ = self.lstm(xb)
            else:
                for lstm_, dropout_ in zip(self.lstm, self.dropouts):
                    xb = lstm_(xb)[0]
            xb = self.linear(xb[:, -1, :])

            return xb.reshape(-1)

    def to_dataset(self, x, y):
        return TensorDataset(torch.Tensor(x).cuda(), torch.Tensor(y).cuda())
