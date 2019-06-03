import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from . import count_parameters

class SASRec(nn.Module):

    def __init__(self, hidden_size, max_seq_len, n_items, dropout=0.2):
        super().__init__()

        '''
        Parameters
        '''
        self.hidden_size = hidden_size      # d
        self.max_seq_len = max_seq_len      # n
        self.n_items = n_items              # |I|
        self.dropout = dropout

        '''
        Positional embedding
        P ~ n x d

        *This doesn't need lookup - it's always added to the sequence
        '''
        self.positional_embedding = torch.FloatTensor(
            self.max_seq_len, hidden_size
        )

        '''
        Item embedding
        M ~ |I| x d
        '''
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size
        )

        '''
        Self-Attention module
        '''
        self.attention_module = CausalSelfAttention(self.hidden_size)

        '''
        Feed-forward network
        '''
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        )

        print('\nCreated "{}" model with {:,} parameters:'.format(
            self.__class__.__name__, count_parameters(self)
        ))
        print('Hidden size {:,}'.format(self.hidden_size))
        print('Max sequence length {:,}'.format(self.max_seq_len))
        print('Item vocabulary {:,}'.format(self.n_items))
        print('[Dropout: {:,.3f}]'.format(self.dropout))

        print(self)

    def forward(self, ixn_seq, verbose=False):
        # E^hat = [M + P]
        input_embedding = self.positional_embedding + self.item_embedding(ixn_seq)
        if verbose:
            print('Input embedding size: {}'.format(input_embedding.size()))

        # Attention
        context, attn_scores = self.attention_module(input_embedding, verbose=verbose)

        # Feed-forward aggregator
        output = self.ffn(context)

        return output

'''
python -m sasrec_torch.model -d 3 -n 10 --data data\\Beauty.txt
'''
if __name__ == "__main__":
    import os
    import argparse

    from datetime import datetime

    from .data import load_sequences

    # Model parser
    parser = argparse.ArgumentParser(description='SASRec Model')
    parser.add_argument('--hidden-size', '-d', type=int, default=256, help='Hidden size')
    parser.add_argument('--max-len', '-n', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--data', type=str, default=os.path.join('data', 'Beauty.txt'),
        help='Location of dataset for training and evaluation')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    args = parser.parse_args()

    # Arguments
    hidden_size = args.hidden_size
    max_seq_len = args.max_len
    dropout = args.dropout
    data_loc = args.data

    # Get data
    start = datetime.now()
    n_users, n_items, train_seq, valid_seq, test_seq = load_sequences(data_loc, as_dict=True)
    pad_ix = n_items
    n_items_w_pad = n_items + 1
    print('{} - Loaded data for splits. Padding index: {}'.format(
        datetime.now() - start, pad_ix
    ))

    model = SASRec(hidden_size, max_seq_len, n_items, dropout)
