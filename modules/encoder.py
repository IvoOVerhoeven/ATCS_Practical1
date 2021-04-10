import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Baseline_Encoder(nn.Module):
    """The baseline encoder.

    """

    def __init__(self, vocab, args):
        super().__init__()

        self.padding_val = vocab['<PAD>']

    def forward(self, embeddings, text):

        mask = (text != self.padding_val).unsqueeze(-1)

        mean_embedding = torch.mean(embeddings * mask, dim=0)

        return mean_embedding


class SimpleLSTM_Encoder(nn.Module):
    """The simple LSTMs encoder.

    """

    def __init__(self, vocab, args):
        super().__init__()

        self.padding_val = vocab['<PAD>']

        self.lstm = nn.LSTM(input_size=vocab.vectors.size(1),
                            hidden_size=args.hidden_dims,
                            bidirectional=args.bidirectional)

    def forward(self, embeddings, text):

        lengths = torch.sum((text != self.padding_val).float(), dim=0).long()

        h_t, _ = self.lstm(embeddings)

        idx = (lengths - 1).view(-1, 1).expand(lengths.size(0), h_t.size(2)).unsqueeze(0)

        out = h_t.gather(0, idx).squeeze()

        return out

class MaxPoolLSTM_Encoder(nn.Module):
    """The maxpooling LSTM encoder.

    """

    def __init__(self, vocab, args):
        super().__init__()

        self.padding_val = vocab['<PAD>']

        self.lstm = nn.LSTM(input_size=vocab.vectors.size(1),
                            hidden_size=args.hidden_dims,
                            bidirectional=args.bidirectional)

    def forward(self, embeddings, text):

        mask = (text != self.padding_val).unsqueeze(-1)

        h_t, _ = self.lstm(embeddings)

        max_pool, _ = torch.max(h_t * mask, dim=0)

        return max_pool
