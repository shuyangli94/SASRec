import torch
import torch.nn as nn
import numpy as np

class SelfAttention(nn.Module):
    """
    Implements Scaled Dot Product attention (Vaswani et al. 2017)
    https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

    Attn(Q, K, V) = softmax(QK^T / sqrt(d)) V
    """
    def __init__(self, hidden_size):
        super().__init__()

        # Hidden size & Scaling
        self.hidden_size = hidden_size
        self.scaling = np.sqrt(self.hidden_size)

        # Projections into the hidden space
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, embedding, mask=None, copy=None, ratings=None):
        # Project the embedding into the key, query, value spaces
        # Embedding: [B; D]
        # Projected: [B; D]
        key = self.key_layer(embedding)
        query = self.query_layer(embedding)
        value = self.value_layer(embedding)

        # Calculate scores
        scores = torch.matmul(query, torch.t(key)) / self.scaling

        # Mask out invalid positions
        # The mask marks valid positions so we invert it using `mask & 0`
        if mask is not None:
            scores.data.masked_fill_(
                mask.unsqueeze(1) == 0, float('-inf')
            )

        # Turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)

        # Copy over existing probability distribution to modify attention weights
        if copy is not None:
            # Can't use += since it's in-place
            alphas = alphas + copy.unsqueeze(1)

            # Normalize to perform a convex combination
            alphas = F.normalize(alphas, p=1, dim=-1)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas
