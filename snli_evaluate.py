import os

import argparse
from distutils.util import strtobool

import torch
import pytorch_lightning as pl
import pickle

from data.snli import SNLI
from models.InferSent import InferSent
from utils.reproducibility import load_latest


def snli_embeddings(args):

    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    model_save_name = "InferSent-" + args.encoder + "_v" + str(args.version)

    model = load_latest(InferSent, model_save_name,
                        inference=True)
    model.eval()
    model.freeze()
    model = model.to(device)

    snli = SNLI()
    snli.prep_data()
    print("Data loaded successfully.'")

    train_loader, valid_loader, test_loader = snli.snli_dataloaders(args.batch_size, device)

    trainer = pl.Trainer(gpus=1 if (torch.cuda.is_available() and args.gpu) else 0)

    train_result = trainer.test(model, test_dataloaders=train_loader, verbose=False)
    val_result = trainer.test(model, test_dataloaders=valid_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"Train": train_result[0]["Test Accuracy"],
              "Test": test_result[0]["Test Accuracy"],
              "Valid": val_result[0]["Test Accuracy"]}

    with open(os.path.join("./checkpoints", model_save_name, f"snli_eval.pkl"), "wb+") as file:
        print(f"Saving to {file.name}", flush=True)
        pickle.dump(result, file)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--encoder', default='Simple', type=str,
                        choices=['Baseline', 'Simple', 'BiSimple', 'BiMaxPool'],
                        help='Which encoder architecture to use. Choose between Baseline, Simple, BiSimple or BiMaxPool.')
    parser.add_argument('--version', default=3, type=int,
                        help='Version number')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for data-loaders')
    parser.add_argument('--gpu', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Whether to evaluate on GPU (if available) or CPU'))

    args = parser.parse_args()

    embeddings = snli_embeddings(args)
