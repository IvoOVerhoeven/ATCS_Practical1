import torch
import numpy as np
from tqdm import tqdm

def generate_embeddings(model, loaders, vocab):

    sent_set = set()
    sentences = []
    embeddings = []

    for data_loader in loaders:
        with tqdm(data_loader, unit="batch") as pbar:
            for batch in pbar:

                premise = batch.premise

                sents = [[vocab.itos[token] for token in sent if (not vocab.itos[token] in vocab.itos[:4])]
                        for sent in premise.T.detach().cpu().numpy()]
                sents = [' '.join(sent) for sent in sents]

                new_idx = [i for i, sent in enumerate(sents) if (not sent in sent_set)]

                premise_ = premise.index_select(dim=1, index=torch.LongTensor(new_idx).to(premise.device))

                with torch.no_grad():
                    encoded = model.encode(premise_).detach().cpu().numpy()

                sent_set.update(sents)
                sentences.extend([sents[ii] for ii in new_idx])
                embeddings.extend(encoded)

                pbar.set_postfix(Representation_length=len(embeddings))

    #sents = [' '.join([vocab.itos[token] for token in sent if (not vocab.itos[token] in vocab.itos[:4])])
    #         for sent in sentences.cpu().numpy()]

    #sents, idx = np.unique(sents, return_index=True)
    embs = torch.stack(embeddings).cpu().numpy()

    return list(zip(sentences, embs))
