import torch
import torch.nn as nn
import torch.functional as F

class InferSent_clf(nn.Module):

    def __init__(self, encoder_output_size, args):
        super().__init__()

        self.Intermediate = nn.Linear(in_features= 4 * encoder_output_size,
                                      out_features=args.linear_dims)

        self.Output = nn.Linear(in_features=args.linear_dims,
                                out_features=args.classes)

    def forward(self, u, v):

        uv_concat = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)

        return self.Output(self.Intermediate(uv_concat))
