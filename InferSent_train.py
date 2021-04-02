import os
import argparse
from distutils.util import strtobool

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import pickle
import tqdm

from utils.reproducibility import set_seed
from utils.timing import Timer
from data.snli import snli_dataloaders
from models.InferSent import InferSent

CHECKPOINT_PATH = './checkpoints'

class LRStop(Exception):
    pass

class LearningRateStopper(pl.Callback):

    def __init__(self, min_lr):
        super().__init__()

        self.min_lr = min_lr

    def on_train_epoch_start(self, trainer, pl_module):
        cur_lr = trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        if cur_lr < self.min_lr:
            print(f'\nLearning rate {cur_lr} dropped below limit {self.min_lr}.\nWill stop training.')
            trainer.should_stop = True

    def on_validation_epoch_end(self, trainer, pl_module):
        cur_lr = trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr']
        trainer.lr_schedulers[0]['scheduler'].optimizer.param_groups[0]['lr'] = pl_module.args.weight_decay * cur_lr


def train(args):
    """
    Trains the classifier.
    Inputs:
        args - Namespace object from the argparser defining the hyperparameters etc.
    """

    full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir + '-' +
                                ('Bi' if args.bidirectional else '' + args.encoder) +
                                f'_v{args.version}')
    os.makedirs(full_log_dir, exist_ok=True)
    os.makedirs(os.path.join(full_log_dir, "lightning_logs"), exist_ok=True)  # to fix "Missing logger folder"
    if args.debug:
        os.makedirs(os.path.join(full_log_dir, "lightning_logs", "debug"), exist_ok=True)

    with open(args.glove_path + "/snli_vocab", "rb") as file:
        vocab = pickle.load(file)

    train_loader, valid_loader, test_loader = snli_dataloaders(args.batch_size,
                                                               snli_path=args.snli_path,
                                                               num_workers=args.num_workers,
                                                               pad_value=vocab['<PAD>'])

    model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="max", monitor="Valid Accuracy")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping = LearningRateStopper(args.min_lr)

    trainer = pl.Trainer(default_root_dir=full_log_dir,
                         checkpoint_callback=model_checkpoint,
                         callbacks=[lr_monitor, early_stopping],
                         gpus=1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_epochs=args.max_epochs,
                         gradient_clip_val=args.max_norm,
                         progress_bar_refresh_rate=args.progress_bar,
                         limit_train_batches=100 if args.debug else 1.0,
                         limit_val_batches=100 if args.debug else 1.0,
                         limit_test_batches=10 if args.debug else 1.0)
                         #fast_dev_run=args.debug)

    trainer.logger._default_hp_metric = None

    if args.debug:
        trainer.logger._version = 'debug'

    set_seed(42)

    model = InferSent(vocab=vocab, args=args)

    timer = Timer()
    trainer.fit(model, train_loader, valid_loader)
    print(f"Total training time: {timer.time()}")

    # Eval post training
    model = InferSent.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test results
    val_result = trainer.test(model, test_dataloaders=valid_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"Test": test_result[0]["Test Accuracy"],
              "Valid": val_result[0]["Test Accuracy"]}

    return model, result

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--encoder', default='Baseline',
                        choices=['Baseline', 'SimpleLSTM', 'MaxPoolLSTM'],
                        help='Which encoder architecture to use')
    parser.add_argument('--glove_path', default='./data/glove',
                        help='Location of GloVe vectors and vocab')
    parser.add_argument('--snli_path', default='./data/snli',
                        help='Location of SNLI data')

    # Module hyperparameters
    ## Embeddings
    parser.add_argument('--embedding_grad', default=False, type=lambda x: bool(strtobool(x)),
                        help='Whether or not GloVe embeddings are trained')

    ## Encoders
    parser.add_argument('--bidirectional', default=False, type=lambda x: bool(strtobool(x)),
                        help=('Whether or not encoder should be bidirectional'))
    parser.add_argument('--hidden_dims', default=1024, type=int,
                        help='Number of hidden nodes in LSTMs.')

    ## Classifier
    parser.add_argument('--linear_dims', default=512, type=int,
                        help='Number of hidden nodes in linear layer classifier.')
    parser.add_argument('--classes', default=3, type=int,
                        help='The number of classes to train on.')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', default=0.99, type=float,
                        help='Lr-decay after every epoch')
    parser.add_argument('--decay_factor', default=0.2, type=float,
                    help='Lr-decay after dev accuracy plateaus')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='Minimum learning rate before stopping training')

    # Trainer hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--max_epochs', default=20, type=int,
                        help='Max number of training epochs')
    parser.add_argument('--max_norm', default=5.0, type=float,
                        help='Gradient clipping value')

    # Other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders.')
    parser.add_argument('--progress_bar', default=10, type=int,
                        help=('How many steps before refreshing progress bar.'))
    parser.add_argument('--log_dir', default='InferSent', type=str,
                        help='Name of the subdirectory for PyTorch Lightning logs and the final model.')
    parser.add_argument('--val_check_prop', default=0.25, type=float,
                        help='How often to check validation performance')

    # Debug parameters
    parser.add_argument('--debug', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Whether to run in debug mode'))
    parser.add_argument('--gpu', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Whether to train on GPU (if available) or CPU'))
    parser.add_argument('--version', default=1, type=int,
                        help='Versioning, because this will take a few iterations.')

    args = parser.parse_args()

    model, results = train(args)

    print(results)
