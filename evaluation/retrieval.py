import torch
import numpy as np
from numpy.linalg import norm

from evaluation.importance_weights import cft_weights
from evaluation.visualization import text_highlighter
from utils.text_processing import text_preprocessor


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

def cft_retrieval_printer(query, representations, model, vocab, top_k=5, max_alpha=0.8, temp=1.0):

    neighbours = highest_sim_retrieval(query=query,
                                       representations=representations,
                                       model=model, vocab=vocab,
                                       top_k=top_k)

    print("Query:")
    text_highlighter(text=query,
                     weights=cft_weights(query, model, vocab),
                     verbose=True,
                     max_alpha=max_alpha,
                     temp=temp)
    print("\nMost similar SNLI sentences:")
    for neighnour in neighbours:
        text_highlighter(text=neighnour,
                         weights=cft_weights(neighnour, model, vocab),
                         verbose=True,
                         max_alpha=max_alpha,
                         temp=temp)
        print("")
