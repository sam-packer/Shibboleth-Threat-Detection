import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.globals import cfg


class BehavioralEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        hidden_layers = cfg("model.architecture.hidden_layers", [64, 48])
        embed_dim = cfg("model.architecture.embed_dim", 32)
        dropout_rate = cfg("model.architecture.dropout", 0.15)

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h

        self.hidden = nn.Sequential(*layers)
        self.fc_out = nn.Linear(prev_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.fc_out(x)
        return F.normalize(x, p=2, dim=1)
