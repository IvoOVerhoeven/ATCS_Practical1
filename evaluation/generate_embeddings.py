import torch
import numpy as np
from tqdm import tqdm

def generate_embeddings(model, loaders, vocab):

    sentences = []
    embeddings = []

    for data_loader in loaders:
        with tqdm(data_loader, unit="batch") as pbar:
            for batch in pbar:

                premise = batch.premise

                with torch.no_grad():
                    encoded = model.encode(premise).detach()

                sentences.extend(premise.detach().T)
                embeddings.extend(encoded)

                pbar.set_postfix(Representation_length=len(embeddings))

    sents = [' '.join([vocab.itos[token] for token in sent if (not vocab.itos[token] in vocab.itos[:4])])
             for sent in sentences.cpu().numpy()]

    sents, idx = np.unique(sents, return_index=True)
    embs = torch.stack(embeddings).cpu().numpy()[idx]

    return list(zip(sents, embs))
