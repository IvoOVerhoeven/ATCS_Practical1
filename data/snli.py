import os
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.vocab import Vocab

from datasets import load_dataset, load_from_disk

import spacy

import pickle

def snli_preprocess(snli_path="./data/snli", glove_path="./data/glove", save=True, verbose=False):
    """
    Preprocesses the SNLI dataset to be compatible with PyTorch.
    Downloads from Huggingface's datasets, tokenizes with Spacy,
    builds GloVe compatible vocab with torchtext and then saves
    everything to disk.

    Args:
        snli_path (str, optional): [description]. Defaults to "./data/snli".
        glove_path (str, optional): [description]. Defaults to "./data/glove".
        save (bool, optional): [description]. Defaults to True.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    #############################################################################
    ##### Load dataset and Spacy model
    #############################################################################
    if verbose:
        print("Loading spacy model.")

    nlp = spacy.load("en_core_web_sm")

    if verbose:
        print("Loading dataset. If not found, will download.")
    dataset = load_dataset('snli')

    # For some reason there appear to be negative labels
    dataset = dataset.filter(lambda example: int(example['label']) in [0, 1, 2])

    #############################################################################
    ##### Construct vocab
    #############################################################################
    if verbose:
        print("Constructing vocab")

    counter = Counter()
    for doc in nlp.tokenizer.pipe(dataset['train']['premise'] + dataset['train']['hypothesis'], batch_size=10000):
        counter.update(list(t.norm_ for t in doc if not t.is_punct))

    vocab = Vocab(counter, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'), specials_first=True)

    if verbose:
        print("Loading GloVe")
    vocab.load_vectors('glove.840B.300d', cache=glove_path, unk_init=torch.Tensor.normal_)

    #############################################################################
    ##### Feed dataset through pre-process pipeline
    #############################################################################
    if verbose:
        print("Preprocessing dataset")

    def preprocess(batch):

        def tokenize(example):
            return list(vocab[t.norm_] for t in nlp.tokenizer(example) if not t.is_punct)

        def _text_transform(example):
            return [vocab['<BOS>']] + tokenize(example) + [vocab['<EOS>']]

        batch['premise'] = _text_transform(batch['premise'])
        batch['hypothesis'] = _text_transform(batch['hypothesis'])
        batch['labels'] = batch['label']

        return batch

    dataset = dataset.map(preprocess, batched=False)
    dataset.set_format(type='torch', columns=['premise', 'hypothesis', 'labels'])

    #############################################################################
    ##### Dump dataset and vocab to ./data directories
    #############################################################################
    if save:
        dataset.save_to_disk(snli_path)

        if verbose:
            print("Saving vocab and processed dataset")

        vocab_file = open(glove_path + "/snli_vocab", "wb")
        pickle.dump(vocab, vocab_file)

    return dataset, vocab


def snli_dataloaders(batch_size, snli_path="./data/snli", num_workers=0, pad_value=3.0, shuffle_train=True):
    """[summary]

    Args:
        batch_size ([type]): [description]
        snli_path (str, optional): [description]. Defaults to "./data/snli".
        glove_path (str, optional): [description]. Defaults to "./data/glove".
        shuffle_train (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """

    dataset = load_from_disk(snli_path)

    def collate_batch(batch):

        premise_list, hypothesis_list, labels_list = [], [], []
        for example in batch:
            premise_list.append(example['premise'])
            hypothesis_list.append(example['hypothesis'])
            labels_list.append(example['labels'])

        output = (pad_sequence(premise_list, padding_value=pad_value),
                pad_sequence(hypothesis_list, padding_value=pad_value),
                torch.stack(labels_list))

        return output

    train_loader = DataLoader(dataset['train'], batch_size=batch_size,
                              shuffle=shuffle_train, collate_fn=collate_batch,
                              num_workers=num_workers, pin_memory=True)

    valid_loader = DataLoader(dataset['validation'], batch_size=batch_size,
                              shuffle=False, collate_fn=collate_batch,
                              num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(dataset['test'], batch_size=batch_size,
                             shuffle=False, collate_fn=collate_batch,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':

    print("Running Preprocessing of SNLI Dataset")
    snli_preprocess(verbose=True)
