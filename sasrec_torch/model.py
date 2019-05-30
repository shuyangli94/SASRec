import torch
import torch.nn as nn

import numpy as np

class SASRec(nn.Module):

    def __init__(self, hidden_size, max_seq_len, n_items):
        super().__init__()

        # Parameters
        self.hidden_size = hidden_size      # d
        self.max_seq_len = max_seq_len      # n
        self.n_items = n_items              # |I|

        # Positional embedding
        # [n; d]
        self.positional_embedding = nn.Embedding(
            self.max_seq_len, hidden_size
        )

        # 