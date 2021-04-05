import torch
import torch.nn as nn
import torch.functional as F

class InferSent_clf(nn.Module):

    def __init__(self, encoder_output_size, args):
        super().__init__()

        self.clf = nn.Sequential(
            nn.Linear(in_features=4 * encoder_output_size, out_features=args.linear_dims),
            nn.Tanh(),
            nn.Linear(in_features=args.linear_dims, out_features=args.linear_dims),
            nn.Tanh(),
            nn.Linear(in_features=args.linear_dims, out_features=args.classes),
        )

    def forward(self, u, v):

        features = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)

        logits = self.clf(features)

        return logits
