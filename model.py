import torch
import torch.nn as nn


class RBAModel(nn.Module):
    def __init__(self, vocab_sizes, embedding_dims, num_numerical_feats, hidden_dim=128, dropout_rate=0.4):
        """
        Initializes a flexible model that can handle any number of categorical and numerical features.

        Args:
            vocab_sizes (dict): A dictionary mapping categorical feature names to their vocabulary size.
                                e.g., {'username': 5000, 'Country': 250}
            embedding_dims (dict): A dictionary mapping feature names to their desired embedding dimension.
                                   Features not in this dict will get a default size of 10.
            num_numerical_feats (int): The number of numerical features.
            hidden_dim (int): The size of the hidden layer.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(RBAModel, self).__init__()

        # --- Create Embedding Layers Dynamically ---
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        for col, size in vocab_sizes.items():
            dim = embedding_dims.get(col, 10)
            self.embeddings[col] = nn.Embedding(size, dim)
            total_embedding_dim += dim

        # --- Fully Connected Layers with Batch Normalization for stability ---
        input_dim = total_embedding_dim + num_numerical_feats

        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)  # Batch Norm Layer

        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Norm Layer

        self.fc3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_categorical, x_numerical):
        """
        Forward pass through the network.

        Args:
            x_categorical (Tensor): A tensor of shape (batch_size, num_categorical_features)
                                    containing integer indices for each categorical feature.
            x_numerical (Tensor): A tensor of shape (batch_size, num_numerical_features)
                                  containing the numerical feature values.
        """
        embedded_outputs = []
        for i, col_name in enumerate(self.embeddings.keys()):
            embedded_outputs.append(self.embeddings[col_name](x_categorical[:, i]))

        # Concatenate all embedding outputs and the numerical features
        x = torch.cat(embedded_outputs + [x_numerical], dim=1)

        # Pass through the fully connected layers with activation, batch norm, and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

