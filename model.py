import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleRBAModel(nn.Module):
    def __init__(self, input_dim: int, num_users: int, embed_dim: int):
        super(SimpleRBAModel, self).__init__()

        # This allows the neural network to learn the behavior for each user
        # Rather than scoring based on general logins, we personalize it for each user
        self.user_embed = nn.Embedding(num_users, embed_dim)

        # 32 -> 16 -> 1 layers
        # The 36 features will stay the same. As we add more rows, these numbers can be increased or potentially more layers added
        self.fc1 = nn.Linear(input_dim + embed_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, user_id: torch.Tensor | None = None) -> torch.Tensor:
        if user_id is None:
            # For an unknown user, use the global login data
            user_vec = torch.zeros((x.size(0), self.user_embed.embedding_dim), device=x.device)
        else:
            # Otherwise, we can personalize based on the user
            user_vec = self.user_embed(user_id)

        # Concatenates features and user embeddings
        x = torch.cat([x, user_vec], dim=1)

        # ReLU activation function for first layer
        x = F.relu(self.fc1(x))
        # Deactivate 20% of nodes on second layer so the neural network doesn't overfit
        x = self.dropout(F.relu(self.fc2(x)))
        # Third layer is the final layer, pass it through a sigmoid to give it a score of 0-1
        x = torch.sigmoid(self.fc3(x))
        return x
