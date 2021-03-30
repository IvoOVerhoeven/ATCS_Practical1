import torch
import torch.nn as nn
import torch.functional as F

class Vocab_Embedding(nn.Module):

    def __init__(self, vocab, args):
        super().__init__()

        self.Embedding = nn.Embedding(num_embeddings=vocab.vectors.size(0),
                                      embedding_dim=vocab.vectors.size(1))
        self.Embedding.weight.data.copy_(vocab.vectors)
        self.Embedding.requires_grad = args.embedding_grad

    def forward(self, input):
        return self.Embedding(input)
