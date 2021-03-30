import torch
import torch.nn as nn
import torch.functional as F

class Baseline_Encoder(nn.Module):
    """The baseline encoder.

    """

    def __init__(self, args):
        super().__init__()

    def forward(self, input):
        return torch.mean(input, dim=0, keepdims=True)


class SimpleLSTM_Encoder(nn.Module):
    """The simple LSTMs encoder.

    """

    def __init__(self, input_size, args):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=args.hidden_dims,
                            bidirectional=args.bidirectional)

    def forward(self, input):

        ht, _ = self.lstm(input)

        return ht[-1]


class MaxPoolLSTM_Encoder(nn.Module):
    """The maxpooling LSTM encoder.

    """

    def __init__(self, input_size, args):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=args.hidden_dims,
                            bidirectional=args.bidirectional)

    def forward(self, input):

        ht, _ = self.lstm(input)

        return torch.max(ht, dim=0)[0]
