import os
import argparse
from distutils.util import strtobool
from collections import defaultdict
import heapq

import torch
import torch.nn.functional as F
import pickle
import numpy as np
import spacy

from data.snli import SNLI
from models.InferSent import InferSent
from utils.reproducibility import load_latest
from evaluation.importance_weights import max_pool_propensity

nlp = spacy.load("en_core_web_sm")

def snli_linguistic_properties(args):

    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    model_save_name = "InferSent-BiMaxPool" + "_v" + str(args.version)  # BiMaxPool

    model = load_latest(InferSent, model_save_name,
                        inference=True)
    model.eval()
    model.freeze()
    model = model.to(device)

    snli = SNLI()
    snli.prep_data()

    vocab = snli.vocab()
    print("Data and vocab loaded successfully.'")

    _, valid_loader, test_loader = snli.snli_dataloaders(args.batch_size, device)

    linguistic_properties = {'pos': defaultdict(list),
                             'tag': defaultdict(list),
                             'dep': defaultdict(list),
                             'iob': defaultdict(list)}
    best_loss = []
    worst_loss = []

    for data_loader in [valid_loader, test_loader]:
        for batch in data_loader:
            with torch.no_grad():
                premise_prop = max_pool_propensity(batch.premise, model)
                hypothesis_prop = max_pool_propensity(batch.hypothesis, model)
                logits = model.forward(batch.premise, batch.hypothesis).detach()
                loss = F.cross_entropy(logits, batch.label, reduction='none').cpu()

            premise_text = [[vocab.itos[token]
                             for token in seq if (not token == vocab["<PAD>"])] for seq in batch.premise.detach().cpu().T]
            hypothesis_text = [[vocab.itos[token] for token in seq if (
                not token == vocab["<PAD>"])] for seq in batch.hypothesis.detach().cpu().T]

            for l, prem, hypo, label in zip(loss.tolist(), premise_text, hypothesis_text, batch.label.tolist()):

                io_tuple = (l, ' '.join(prem[1:-1]), ' '.join(hypo[1:-1]), label)
                if len(worst_loss) < 99:
                    heapq.heappush(worst_loss, io_tuple)
                else:
                    heapq.heapreplace(worst_loss, io_tuple)

                io_tuple = (-l, ' '.join(prem[1:-1]), ' '.join(hypo[1:-1]), label)
                if len(best_loss) < 99:
                    heapq.heappush(best_loss, io_tuple)
                else:
                    heapq.heapreplace(best_loss, io_tuple)

            for text, weights in zip(premise_text + hypothesis_text, premise_prop + hypothesis_prop):
                w_ = weights[1:]
                if len(text) > len(w_): continue
                for i, token in enumerate(nlp(' '.join(text[1:-1]))):
                    linguistic_properties['pos'][token.pos_].append(w_[i])
                    linguistic_properties['tag'][token.tag_].append(w_[i])
                    linguistic_properties['dep'][token.dep_].append(w_[i])
                    linguistic_properties['iob'][token.ent_iob_].append(w_[i])

    linguistic_properties_summary = {'pos': defaultdict(),
                                     'tag': defaultdict(),
                                     'dep': defaultdict(),
                                     'iob': defaultdict()}
    for k in linguistic_properties.keys():
        for kk in linguistic_properties[k].keys():
            data = linguistic_properties[k][kk]
            summary = {'N': len(data), 'mean': np.mean(data),
                    'std': np.std(data),
                    'quantiles': (np.quantile(data, 0.00), np.quantile(data, 0.10),
                                    np.quantile(data, 0.25), np.quantile(data, 0.50),
                                    np.quantile(data, 0.75), np.quantile(data, 0.90),
                                    np.quantile(data, 1.00))}
            linguistic_properties_summary[k][kk] = summary

    with open(os.path.join("./checkpoints", model_save_name, f"best_worst_loss.pkl"), "wb+") as file:
        print(f"Saving to {file.name}", flush=True)
        pickle.dump({'best': heapq.nsmallest(len(best_loss)-1, best_loss),
                    'worst': heapq.nlargest(len(worst_loss)-1, worst_loss)}, file)

    with open(os.path.join("./checkpoints", model_save_name, f"linguistic_properties_summary.pkl"), "wb+") as file:
        print(f"Saving to {file.name}", flush=True)
        pickle.dump(linguistic_properties_summary, file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--version', default=3, type=int,
                        help='Version number')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for data-loaders')
    parser.add_argument('--gpu', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Whether to evaluate on GPU (if available) or CPU'))

    args = parser.parse_args()

    embeddings = snli_linguistic_properties(args)

