import os

import torch
import torchtext
import torchtext.legacy as legacy
import pytorch_lightning as pl

class SNLI():

    def __init__(self, root="./data", glove_path="./data/glove"):

        self.root = root
        self.glove_path = glove_path

    def prep_data(self):
        self.TEXT = legacy.data.Field(sequential=True,
                                      init_token='<BOS>',
                                      eos_token='<EOS>',
                                      lower=True,
                                      tokenize='spacy',
                                      tokenizer_language='en_core_web_sm',
                                      pad_token='<PAD>',
                                      unk_token='<UNK>')

        self.LABEL = legacy.data.Field(sequential=False,
                                       is_target=True,
                                       unk_token=None)

        self.train, self.valid, self.test = legacy.datasets.SNLI.splits(self.TEXT,
                                                                        self.LABEL,
                                                                        root=self.root)

        self.TEXT.build_vocab(self.train, self.valid, self.test, specials_first=True)

        vocab_fp = os.path.join(self.glove_path, "snli_vocab.pt")
        if os.path.exists(vocab_fp):
            print(f"Using dictionary with GloVe found at {vocab_fp}")
            self.TEXT.vocab = torch.load(os.path.join(self.glove_path, "snli_vocab.pt"))
        else:
            print(f"Building dictionary with GloVe-840B-300D.")
            self.TEXT.vocab.load_vectors('glove.840B.300d', cache=self.glove_path)
            torch.save(self.TEXT.vocab, os.path.join(self.glove_path, "snli_vocab.pt"))

        self.LABEL.build_vocab(self.train, specials_first=True)


    def snli_dataloaders(self, batch_size, device):

        train_loader = torchtext.legacy.data.Iterator(dataset=self.train,
                                                      batch_size=batch_size,
                                                      train=True,
                                                      shuffle=True,
                                                      device=device)

        valid_loader = torchtext.legacy.data.Iterator(dataset=self.valid,
                                                      batch_size=batch_size,
                                                      train=False,
                                                      shuffle=False,
                                                      device=device)

        test_loader = torchtext.legacy.data.Iterator(dataset=self.test,
                                                     batch_size=batch_size,
                                                     train=False,
                                                     shuffle=False,
                                                     device=device)

        return train_loader, valid_loader, test_loader

    def vocab(self):
        return self.TEXT.vocab

    def label_map(self):
        return self.LABEL.stoi
