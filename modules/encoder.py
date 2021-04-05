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

        mean_embedding = torch.mean(embeddings * (text != self.padding_val)[:, :, None], dim=0)

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

        lengths = torch.sum((text != self.padding_val).float(), dim=0).long().cpu()

        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False)

        out, _ = self.lstm(packed)

        seq_unpacked, lens_unpacked = pad_packed_sequence(out)

        # https://blog.nelsonliu.me/2018/01/25/extracting-last-timestep-outputs-from-pytorch-rnns/
        idx = (torch.LongTensor(lens_unpacked) -
               1).view(-1, 1).expand(len(lengths), seq_unpacked.size(2))
        idx = idx.unsqueeze(0)

        out_T = seq_unpacked.gather(0, idx.to(embeddings.device)).squeeze(0)

        return out_T


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

        out, _ = self.lstm(embeddings)

        mask = (text != self.padding_val).to(embeddings.device)
        max_pool = torch.max(out * mask[:, :, None], dim=0)[0]

        return max_pool
