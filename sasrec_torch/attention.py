import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalSelfAttention(nn.Module):
    """
    Implements Scaled Dot Product attention (Vaswani et al. 2017)
    https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

    Includes causality, etc. from (Kang & McAuley 2018):
    https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf

    Attn(Q, K, V) = softmax(QK^T / sqrt(d)) V
    """
    def __init__(self, hidden_size, causality=True, dropout=0.2):
        super().__init__()

        # Hidden size & Scaling
        self.hidden_size = hidden_size
        self.scaling = np.sqrt(self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        # Projections into the hidden space
        self.key_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        # Causality (Future Blinding)
        self.causality = causality

    def forward(self, embedding, mask=None, copy=None, verbose=False):
        # Project the embedding into the key, query, value spaces
        # Embedding: [B; D]
        # Projected: [B; D]
        key = self.key_layer(embedding)
        query = self.query_layer(embedding)
        value = self.value_layer(embedding)
        if verbose:
            print('Key size: {}'.format(key.size()))
            print('Query size: {}'.format(query.size()))
            print('Value size: {}'.format(value.size()))

        # Calculate scores
        scores = torch.matmul(query, torch.t(key)) / self.scaling
        if verbose:
            print('Scores size: {}'.format(scores.size()))

        # Mask out invalid positions
        # The mask marks valid positions so we invert it using `mask & 0`
        if mask is not None:
            if verbose:
                print('Mask size: {}'.format(mask.size()))
            scores.data.masked_fill_(
                mask.unsqueeze(1) == 0, float('-inf')
            )

        # Turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)
        if verbose:
            print('Alphas size: {}'.format(alphas.size()))

        # Copy over existing probability distribution to modify attention weights
        if copy is not None:
            # Can't use += since it's in-place
            alphas = alphas + copy.unsqueeze(1)

            # Normalize to perform a convex combination
            alphas = F.normalize(alphas, p=1, dim=-1)

        # Dropout
        alphas = self.dropout(alphas)

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        if verbose:
            print('Context size: {}'.format(context.size()))

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas
