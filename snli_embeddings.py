import os

import argparse
from distutils.util import strtobool

import torch
import pickle

from data.snli import SNLI
from models.InferSent import InferSent
from utils.reproducibility import load_latest
from evaluation.generate_embeddings import generate_embeddings


def snli_embeddings(args):

    model_save_name = "InferSent-" + args.encoder + "_v" + str(args.version)

    model = load_latest(InferSent, model_save_name,
                        inference=True,
                        map_location='cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
    model.eval()
    model.freeze()

    snli = SNLI()
    snli.prep_data()

    vocab = snli.vocab()
    print("Data and vocab loaded successfully.'")

    train_loader, valid_loader, test_loader = snli.snli_dataloaders(args.batch_size, model.device)

    embeddings = generate_embeddings(model,
                                     [train_loader, valid_loader, test_loader],
                                     vocab)

    with open(os.path.join("./checkpoints", model_save_name, f"snli_embeddings.pkl"), "wb+") as file:
        print(f"Saving to {file.name}", flush=True)
        pickle.dump(embeddings, file)

    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--encoder', default='Simple', type=str,
                        choices=['Baseline', 'Simple', 'BiSimple', 'BiMaxPool'],
                        help='Which encoder architecture to use. Choose between Baseline, Simple, BiSimple or BiMaxPool.')
    parser.add_argument('--version', default=3, type=int,
                        help='Version number')

    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size for data-loaders')
    parser.add_argument('--gpu', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Whether to evaluate on GPU (if available) or CPU'))

    args = parser.parse_args()

    embeddings = snli_embeddings(args)
