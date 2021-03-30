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

        self.args = args

        self.embedding = Vocab_Embedding(vocab, args)

        if args.encoder == 'Baseline':
            self.encoder = MaxPoolLSTM_Encoder(input_size=vocab.vectors.size(1), args=args)
        elif args.encoder == 'SimpleLSTM':
            self.encoder = MaxPoolLSTM_Encoder(input_size=vocab.vectors.size(1), args=args)
        elif args.encoder == 'MaxPoolLSTM':
            self.encoder = MaxPoolLSTM_Encoder(input_size=vocab.vectors.size(1), args=args)

        encoder_output_dims = (2 if args.bidirectional else 1) * args.hidden_dims
        self.classifier = InferSent_clf(encoder_output_dims, args)

        # TODO: does this need to ignore padding?
        self.loss_module = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    def encode(self, text):
        embedding = self.embedding(text)
        encoded = self.encoder(embedding)

        return encoded

    def forward(self, premise, hypothesis):
        u = self.encode(premise)
        v = self.encode(hypothesis)

        return self.classifier(u, v)

    def training_step(self, batch, batch_idx):
        premise, hypothesis, labels = batch

        logits = self.forward(premise, hypothesis)

        loss = self.loss_module(logits, labels)

        self.log('Train CE_Loss', loss, on_step=True)

        with torch.no_grad():
            acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())

        self.log('Train Accuracy', acc, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        premise, hypothesis, labels = batch

        logits = self.forward(premise, hypothesis)

        loss = self.loss_module(logits, labels)

        self.log('Valid CE_Loss', loss, on_epoch=True)

        acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())

        self.log('Valid Accuracy', acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        premise, hypothesis, labels = batch

        logits = self.forward(premise, hypothesis)

        loss = self.loss_module(logits, labels)

        self.log('Test CE_Loss', loss, on_epoch=True)

        acc = torch.mean((torch.argmax(logits, dim=-1) == labels).float())

        self.log('Test Accuracy', acc, on_epoch=True)

    def configure_optimizers(self):

        optimizer = optim.SGD([{'params': self.encoder.parameters()}, {'params': self.classifier.parameters()}],
                              lr=self.args.lr, weight_decay=self.args.weight_decay)

        lr_scheduler  = {'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='max',
                                                                factor=self.args.decay_factor,
                                                                patience=1,
                                                                min_lr=self.args.min_lr),
                            'monitor': 'Valid Accuracy',
                            'name': 'Learning Rate'
                        }

        return [optimizer], [lr_scheduler]
