from collections import Counter

import numpy as np
from numpy.linalg import norm
import torch
import torch.nn.functional as F

from utils.text_processing import text_preprocessor
from evaluation.visualization import text_highlighter

def cft_weights(text, model, vocab):

    toks = text_preprocessor(text, vocab)

    text_torch = torch.tensor(toks).long()

    text_cft = text_torch.detach().clone().tile((len(toks), 1))
    text_cft[np.diag_indices(text_cft.shape[0])] = 0

    text_torch_emb = model.encode(text_torch).tile((len(toks), 1))
    text_cft_emb = model.encode(text_cft)

    cft_sims = F.cosine_similarity(text_torch_emb, text_cft_emb)[1:-1].numpy()

    return 1 - cft_sims


def max_pool_propensity(text_processed, model):
    """Method for computing the max-pool relative frequency given input embedding and model.

    Args:
        text_processed ([type]): [description]
        model ([type]): [description]

    Returns:
        prop: array of scores
    """

    mask = (text_processed != model.encoder.padding_val).unsqueeze(-1)

    embs = model.embedding(text_processed)
    h_t, _ = model.encoder.lstm(embs)

    idx = torch.max(h_t * mask, dim=0)[1].tolist()
    counts = [Counter(id) for id in idx]

    prop = [[c[key] / h_t.size(-1) for key in c.keys()] for c in counts]
    prop = [[val / (1 / len(p)) for val in p] for p in prop]

    return prop

