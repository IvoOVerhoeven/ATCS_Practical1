from __future__ import absolute_import, division, unicode_literals

import os
import sys
import logging
import argparse
from distutils.util import strtobool

from itertools import zip_longest

import torch
import pickle
import sklearn

from models.InferSent import InferSent
from modules.embedding import Vocab_Embedding
from utils.text_processing import text_preprocessor, vocab_builder
from utils.reproducibility import load_latest

def prepare(params, samples):

    params.vocab = vocab_builder(samples)

    params.InferSent.embedding = Vocab_Embedding(params.vocab, None).to(params.InferSent.device)


def batcher(params, batch):

    sentences = [text_preprocessor(s, params.vocab) for s in batch]

    sentences = list(zip_longest(*sentences, fillvalue=params.vocab["<PAD>"]))

    sentences = torch.LongTensor(sentences).to(params.InferSent.device)

    with torch.no_grad():
        embeddings = params.InferSent.encode(sentences)

    return embeddings.detach().cpu().numpy()

def eval(args):

    data_path = args.senteval_path + '/data'
    print(f"Data loc: {data_path}", flush=True)

    sys.path.insert(0, args.senteval_path)

    import senteval

    model_save_name = "InferSent-" + args.encoder + "_v" + str(args.version)

    if args.config == "fast":
        params = {'task_path': data_path, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    elif args.config == "default":
        params = {'task_path': data_path, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}
    else:
        raise ValueError(f"{args.config} is not a recognized SentEval profile. Please choose either 'fast' or 'default'.")

    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    model = load_latest(InferSent, save_name=model_save_name,
                        inference=True)
    model.eval()
    model.freeze()
    model = model.to(device)

    params['InferSent'] = model

    se = senteval.engine.SE(params, batcher, prepare)

    if args.tasks == 'all':
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                          'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                          'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                          'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                          'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'
                          ]
    elif args.tasks == 'infersent':
        transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC',
                          'SICKRelatedness', 'SICKEntailment', 'STS14'
                          ]
    elif args.tasks == 'coco':
        transfer_tasks = ['ImageCaptionRetrieval'
                          ]
    elif args.tasks == 'transfer_all':
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC', 'SNLI',
                          'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                          'STS12', 'STS13', 'STS14', 'STS15', 'STS16'
                          ]
    elif args.tasks == 'probing_all':
        transfer_tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                          'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion'
                          ]
    elif args.tasks == 'test':
        transfer_tasks = ['TREC']
    else:
        raise ValueError(
            f"{args.tasks} is not a recognized SentEval args.tasks subset. Please choose either"+\
                "'all', 'infersent', 'working', 'transfer_all' or 'probing_all'.")

    print("Evaluating on:")
    print(transfer_tasks)

    results = se.eval(transfer_tasks)

    with open(os.path.join("./checkpoints", model_save_name, f"SentEval_{args.config}_{args.tasks}.pkl"), "wb+") as file:
        print("Saving to " + file.name, flush=True)
        pickle.dump(results, file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--senteval_path', default='../SentEval', type=str,
                        help='Path to SentEval package. Typically Users/usr/SentEval for CPU, /home/usr/SentEval for LISA.')

    parser.add_argument('--encoder', default='Baseline', type=str,
                        choices=['Baseline', 'Simple', 'BiSimple', 'BiMaxPool'],
                        help='Which encoder architecture to use. Choose between Baseline, Simple, BiSimple or BiMaxPool.')
    parser.add_argument('--version', default=3, type=int,
                        help='Version number')

    parser.add_argument('--config', default='default', type=str,
                        choices=['default', 'fast'],
                        help='Which classifier configuration to use.')
    parser.add_argument('--tasks', default='infersent_ext', type=str,
                        choices=['all', 'infersent', 'infersent_ext', 'transfer_all', 'probing_all', 'test'],
                        help='Which task profile to use. Option "infersent" matches original paper.')

    parser.add_argument('--gpu', default=True, type=lambda x: bool(strtobool(x)),
                    help=('Whether to evaluate on GPU (if available) or CPU'))

    args = parser.parse_args()

    results = eval(args)

    print(results)
