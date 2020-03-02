from torch.autograd import Variable
import torch
import torch.nn as nn

""" file that implements LSTMs with various recurrent dropout """

class LSTM_Gal(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0, batch_first=True):
        """Initialize params."""
        super(LSTM_Gal, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = dropout

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.input_in = nn.Linear(input_size, hidden_size)
        self.input_forget = nn.Linear(input_size, hidden_size)
        self.input_cell = nn.Linear(input_size, hidden_size)
        self.input_out = nn.Linear(input_size, hidden_size)

        self.hidden_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_forget = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_cell = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self._input_dropout_mask = self._h_dropout_mask = None

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        # tag = None  #
        if hidden is None:
            hidden = self._init_hidden(input)

        def recurrence(input, hidden):
            """Recurrence helper."""

            if self._input_dropout_mask is None:
                self.set_dropout_masks(input.size(0))

            hx, cx = hidden  # n_b x hidden_dim

            # gates = self.input_weights(input) + self.hidden_weights(hx)
            # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            try:
                ingate = self.input_in(input * self._input_dropout_mask[0]) + self.hidden_in(hx * self._h_dropout_mask[0])
            except:
                pass
            forgetgate = self.input_forget(input * self._input_dropout_mask[1]) + self.hidden_forget(
                hx * self._h_dropout_mask[1])
            cellgate = self.input_cell(input * self._input_dropout_mask[2]) + self.hidden_cell(
                hx * self._h_dropout_mask[2])
            outgate = self.input_out(input * self._input_dropout_mask[3]) + self.hidden_out(
                hx * self._h_dropout_mask[3])

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)

            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        self._input_dropout_mask = None
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

    def _init_hidden(self, input_):
        # h = torch.zeros_like(input_.reshape(1, input_.size(1), -1))
        # c = torch.zeros_like(input_.reshape(1, input_.size(1), -1))
        h = torch.zeros(input_.size(0), self.hidden_size).cuda()
        c = torch.zeros(input_.size(0), self.hidden_size).cuda()
        return h, c

    def set_dropout_masks(self, batch_size):
        if self.dropout:
            if self.training:
                self._input_dropout_mask = Variable(torch.bernoulli(
                    torch.Tensor(4, batch_size, self.input_size).fill_(1 - self.dropout)),
                    requires_grad=False)
                self._h_dropout_mask = Variable(torch.bernoulli(
                    torch.Tensor(4, batch_size, self.hidden_size).fill_(1 - self.dropout)),
                    requires_grad=False)

                if torch.cuda.is_available():
                    self._input_dropout_mask = self._input_dropout_mask.cuda()
                    self._h_dropout_mask = self._h_dropout_mask.cuda()
            else:
                self._input_dropout_mask = self._h_dropout_mask = [1. - self.dropout] * 4
        else:
            self._input_dropout_mask = self._h_dropout_mask = [1.] * 4

class LSTM_Semenuita(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0.0, batch_first=True):
        """Initialize params."""
        super(LSTM_Semenuita, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first
        self.dropout = dropout

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.input_in = nn.Linear(input_size, hidden_size)
        self.input_forget = nn.Linear(input_size, hidden_size)
        self.input_cell = nn.Linear(input_size, hidden_size)
        self.input_out = nn.Linear(input_size, hidden_size)

        #TODO remove bias=False ?
        self.hidden_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_forget = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_cell = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self._input_dropout_mask = self._h_dropout_mask = None

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        # tag = None  #
        if hidden is None:
            hidden = self._init_hidden(input)

        def recurrence(input, hidden):
            """Recurrence helper."""

            self.set_dropout_masks(input.size(0))

            hx, cx = hidden  # n_b x hidden_dim

            # gates = self.input_weights(input) + self.hidden_weights(hx)
            # ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            # ingate = self.input_in(input * self._input_dropout_mask[0]) + self.hidden_in(hx * self._h_dropout_mask[0])
            # forgetgate = self.input_forget(input * self._input_dropout_mask[1]) + self.hidden_forget(
            #     hx * self._h_dropout_mask[1])
            # cellgate = self.input_cell(input * self._input_dropout_mask[2]) + self.hidden_cell(
            #     hx * self._h_dropout_mask[2])
            # outgate = self.input_out(input * self._input_dropout_mask[3]) + self.hidden_out(
            #     hx * self._h_dropout_mask[3])

            ingate = self.input_in(input) + self.hidden_in(hx)
            forgetgate = self.input_forget(input) + self.hidden_forget(hx)
            cellgate = self.input_cell(input) + self.hidden_cell(hx)
            outgate = self.input_out(input) + self.hidden_out(hx)

            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)  # o_t
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate * self.drop_g_mask)
            hy = outgate * torch.tanh(cy)  # n_b x hidden_dim

            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

            # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
            # output.append(isinstance(hidden, tuple) and hidden[0] or hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden

    # @staticmethod
    def _init_hidden(self,input_):
        # h = torch.zeros_like(input_.reshape(1, input_.size(1), -1))
        # c = torch.zeros_like(input_.reshape(1, input_.size(1), -1))
        h = torch.zeros(input_.size(0), self.hidden_size).cuda()
        c = torch.zeros(input_.size(0), self.hidden_size).cuda()
        return h, c

    def set_dropout_masks(self, batch_size):
        if self.dropout:
            if self.training:
                self.drop_g_mask = Variable(
                    torch.bernoulli(torch.Tensor(batch_size, self.hidden_size).fill_(1 - self.dropout)),
                    requires_grad=False)

                # self._input_dropout_mask = Variable(torch.bernoulli(
                #     torch.Tensor(batch_size, self.input_size).fill_(1 - self.dropout)), requires_grad=False)
                # self._h_dropout_mask = Variable(torch.bernoulli(
                #     torch.Tensor(batch_size, self.hidden_size).fill_(1 - self.dropout)), requires_grad=False)

                if torch.cuda.is_available():
                    self.drop_g_mask = self.drop_g_mask.cuda()
                    # self._input_dropout_mask = self._input_dropout_mask.cuda()
                    # self._h_dropout_mask = self._h_dropout_mask.cuda()
            else:
                self.drop_g_mask = 1. - self.dropout
        else:
            self.drop_g_mask = 1.
