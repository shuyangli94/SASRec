import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from . import count_parameters

class SASRec(nn.Module):

    def __init__(self, positional_embedding, hidden_size, max_seq_len, n_items, dropout=0.2):
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
        self.positional_embedding = positional_embedding

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

    def forward(self, user, ixn_seq, pos_seq, neg_seq, verbose=False):
        # E^hat = [M + P]
        item_emb = self.item_embedding(ixn_seq)
        input_embedding = self.positional_embedding + item_emb
        if verbose:
            print('Input embedding size: {}'.format(input_embedding.size()))

        # Attention
        context, attn_scores = self.attention_module(input_embedding, verbose=verbose)

        # Feed-forward aggregator
        sas_attn = self.ffn(context)
        if verbose:
            print('SAS scores (F_t^(i)) size: {}'.format(sas_attn.size()))

        # MF layer
        pos_emb = self.item_embedding(pos_seq)
        if verbose:
            print('Positive embedding size: {}'.format(pos_emb.size()))
        relevance_pos = torch.matmul(pos_emb, sas_attn)
        if verbose:
            print('Positive relevance scores size: {}'.format(relevance_pos.size()))
        neg_emb = self.item_embedding(neg_seq)
        if verbose:
            print('Negative embedding size: {}'.format(neg_emb.size()))
        relevance_neg = torch.matmul(neg_emb, sas_attn)
        if verbose:
            print('Negative relevance scores size: {}'.format(relevance_neg.size()))

        return relevance_pos, relevance_neg

'''
python -m sasrec_torch.model -d 3 -n 10 --data data\\Beauty.txt
'''
if __name__ == "__main__":
    import os
    import sys
    import argparse

    from datetime import datetime
    from tqdm import tqdm

    from . import get_device
    from .data import load_sequences, WarpSampler

    USE_CUDA, DEVICE = get_device()

    # Model parser
    parser = argparse.ArgumentParser(description='SASRec Model')

    # Data arguments
    parser.add_argument('--data', type=str, default=os.path.join('data', 'Beauty.txt'),
        help='Location of dataset for training and evaluation')
    parser.add_argument('--workers', type=int, default=3, help='Number of workers')

    # Training hyperparameters
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max-epochs', '-e', type=int, default=201, help='Maximum # epochs trained')

    # Model specificatinos
    parser.add_argument('--hidden-size', '-d', type=int, default=256, help='Hidden size')
    parser.add_argument('--max-len', '-n', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    args = parser.parse_args()

    # Arguments
    data_loc = args.data
    n_workers = args.workers
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    hidden_size = args.hidden_size
    max_seq_len = args.max_len
    dropout = args.dropout

    # Get data
    start = datetime.now()
    n_users, n_items, train_seq, valid_seq, test_seq = load_sequences(data_loc, as_dict=True)
    pad_ix = n_items
    n_items_w_pad = n_items + 1
    print('{} - Loaded data for splits. Padding index: {}'.format(
        datetime.now() - start, pad_ix
    ))

    # Logging stats
    print('Training: Longest sequence {:,} actions, average {:,.2f} actions'.format(
        max([len(seq) for seq in train_seq.values()]),
        sum([len(seq) for seq in train_seq.values()]) / len(train_seq)
    ))

    # Make the model
    positional_emb = torch.FloatTensor(
        max_seq_len, hidden_size
    ).to(DEVICE)
    model = SASRec(
        positional_embedding=positional_emb,
        hidden_size=hidden_size,
        max_seq_len=max_seq_len,
        n_items=n_items,
        dropout=dropout,
    )
    if USE_CUDA:
        model.cuda()

    # Make sampler
    sampler = WarpSampler(
        user_sequences=train_seq,
        n_users=n_users,
        n_items=n_items,
        pad_ix=pad_ix,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_workers=n_workers
    )

    # Training
    n_batches = int(len(train_seq) / batch_size)
    print('{} - Training with {:,} batches of size {:,}, over max {:,} epochs, with LR {:.4f}'.format(
        datetime.now() - start,
        n_batches,
        batch_size,
        max_epochs,
        learning_rate
    ))

    # TRAIN
    try:
        for epoch in range(1, max_epochs + 1):
            for step in tqdm(range(n_batches), total=n_batches, leave=False, unit='b'):
                batch = sampler.next_batch()
                u, seq, pos, neg = [torch.LongTensor(t).to(DEVICE) for t in batch]
                a, b = model.forward(u, seq, pos, neg, verbose=True)
                break
    except Exception as e:
        sampler.close()
        raise
