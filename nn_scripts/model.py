import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers.globals import CONFIG


class SimpleRBAModel(nn.Module):
    def __init__(self, input_dim: int, num_users: int, embed_dim: int):
        super().__init__()

        arch_cfg = CONFIG["model"]["architecture"]

        hidden_layers = arch_cfg.get("hidden_layers", [32, 16])
        dropout_rate = arch_cfg.get("dropout", 0.2)

        # This allows the neural network to learn the behavior for each user
        # Rather than scoring based on general logins, we personalize it for each user
        self.user_embed = nn.Embedding(num_users, embed_dim)

        # Built our layers dynamically from the configuration file
        layers = []
        prev_dim = input_dim + embed_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            prev_dim = h

        self.layers = nn.ModuleList(layers)
        self.fc_out = nn.Linear(prev_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, user_id: torch.Tensor | None = None) -> torch.Tensor:
        if user_id is None:
            # For an unknown user, use the global login data
            user_vec = torch.zeros((x.size(0), self.user_embed.embedding_dim), device=x.device)
        else:
            # Otherwise, we can personalize based on the user
            user_vec = self.user_embed(user_id)

        # Concatenates features and user embeddings
        x = torch.cat([x, user_vec], dim=1)

        # apply hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = torch.sigmoid(self.fc_out(x))
        return x
