import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from modules.embedding import Vocab_Embedding
from modules.encoder import Baseline_Encoder, SimpleLSTM_Encoder, MaxPoolLSTM_Encoder
from modules.classifier import InferSent_clf

class InferSent(pl.LightningModule):

    def __init__(self, vocab, args):
        super().__init__()

        self.save_hyperparameters()

        self.args = args

        self.embedding = Vocab_Embedding(vocab, args)
        if (not args.embedding_grad): self.embedding.freeze()

        if args.encoder == 'Baseline':
            self.encoder = Baseline_Encoder(vocab, args)
        elif args.encoder == 'Simple':
            self.encoder = SimpleLSTM_Encoder(vocab, args)
        elif args.encoder == 'MaxPool':
            self.encoder = MaxPoolLSTM_Encoder(vocab, args)

        if args.encoder == 'Baseline':
            encoder_output_dims = vocab.vectors.size(1)
        else:
            encoder_output_dims = (2 if args.bidirectional else 1) * args.hidden_dims

        self.classifier = InferSent_clf(encoder_output_dims, args)

        # TODO: does this need to ignore padding?
        self.loss_module = nn.CrossEntropyLoss()

    def encode(self, text):

        embedding = self.embedding(text)

        encoded = self.encoder(embedding, text)

        return encoded

    def forward(self, premise, hypothesis):

        u = self.encode(premise)

        v = self.encode(hypothesis)

        logits = self.classifier(u, v)

        return logits

    def training_step(self, batch, batch_idx):

        premise, hypothesis, labels = batch.premise, batch.hypothesis, batch.label

        logits = self.forward(premise, hypothesis)

        loss = self.loss_module(logits, labels)

        self.log('Train CE_Loss', loss, on_step=True)

        with torch.no_grad():
            acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())

        self.log('Train Accuracy', acc, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        premise, hypothesis, labels = batch.premise, batch.hypothesis, batch.label

        logits = self.forward(premise, hypothesis)

        loss = self.loss_module(logits, labels)

        self.log('Valid CE_Loss', loss, on_epoch=True)

        acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())

        self.log('Valid Accuracy', acc, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        premise, hypothesis, labels = batch.premise, batch.hypothesis, batch.label

        logits = self.forward(premise, hypothesis)

        loss = self.loss_module(logits, labels)

        self.log('Test CE_Loss', loss, on_epoch=True)

        acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())

        self.log('Test Accuracy', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):

        optimizer = optim.SGD(self.parameters(),
                              lr=self.args.lr)

        plateau_scheduler  = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='max',
                                                                factor=self.args.decay_factor,
                                                                patience=0,
                                                                cooldown=0,
                                                                verbose=True),
                               'reduce_on_plateau': True,
                               'monitor': 'Valid Accuracy',
                               'interval': 'epoch',
                               'name': 'LR Reduce on Plateau'
                        }

        return [optimizer], [plateau_scheduler]
