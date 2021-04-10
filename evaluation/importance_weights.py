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


def highest_sim_retrieval(query, representations, model, vocab, top_k=5):

    def _cos_sim(x1, x2): return np.dot(x1, x2) / (norm(x1) * norm(x2))
    def _euc_sim(x1, x2): return norm(x1 - x2)

    query_ = text_preprocessor(query, vocab)

    query_embedding = model.encode(torch.tensor(query_)).numpy()

    sims = np.array([_cos_sim(query_embedding, embed) for _, embed in representations])

    idx = np.argsort(sims)[::-1]

    nearest_sim_scores, nearest_neighbours = [], []

    for i in idx:
        if len(nearest_neighbours) >= top_k:
            break

        if (not representations[i][0] in nearest_neighbours):
            nearest_neighbours.append(representations[i][0])
            nearest_sim_scores.append(sims[i])

    return nearest_neighbours
