import os
import random
import sys

import numpy as np
import torch
import pytorch_lightning as pl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def set_deterministic():
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
