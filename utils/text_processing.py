from collections import Counter

import torchtext
from torchtext.vocab import Vocab
import spacy
nlp = spacy.load("en_core_web_sm")


def text_tokenizer(text, tokenizer=nlp.tokenizer):
    return [token.norm_ for token in tokenizer(str(text))]


def text_preprocessor(text, vocab, tokenizer=nlp.tokenizer):

    text_ = [token.norm_ for token in tokenizer(str(text))]
    toks = [vocab["<BOS>"]] + [vocab[q] if vocab[q] != None else vocab["UNK"] for q in text_] + [vocab["<EOS>"]]

    return toks


def vocab_builder(corpus, glove_path="./data/glove"):

    corpus_counter = Counter()
    for sent in corpus:
        corpus_counter.update(text_tokenizer(sent))

    vocab = Vocab(corpus_counter, specials=("<UNK>", "<PAD>", "<BOS>", "<EOS>"))
    vocab.load_vectors("glove.840B.300d", cache=glove_path)

    return vocab
