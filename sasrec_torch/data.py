import random
import numpy as np
import pandas as pd

from multiprocessing import Process, Queue

def random_sample_item(n_items, excluded):
    """
    Randomly sample an item index, given a list of excluded items
    
    Arguments:
        n_items {int} -- Total # items
        excluded {set} -- Set of items not able to be sampled
    
    Returns:
        int -- Randomly sampled item
    """
    sample = np.random.randint(0, n_items)
    while sample in excluded:
        sample = np.random.randint(0, n_items)
    return sample

def sample_sequence(user_sequences, n_users, n_items, max_seq_len, pad_ix):
    """
    Samples a sequence of last [max_seq_len] items a user has interacted with
    
    Arguments:
        user_sequences {map} -- user_sequences[u] is the interaction sequence for a user
        n_users {int} -- Number of unique users
        n_items {int} -- Number of unique items
        max_seq_len {int} -- Maximum length of an interaction sequence
        pad_ix {int} -- Index of padding item
    
    Returns:
        np.array -- Sequence of visible items
        np.array -- Sequence of positive items (next predicted)
        np.array -- Sequence of negative items
    """
    # Sample a user who has an interaction sequence
    user = np.random.randint(0, n_users)
    while len(user_sequences[user]) <= 1:
        user = np.random.randint(0, n_users)

    # Sequence, positives for next-item training, negatives, and final item
    sequence = np.zeros([max_seq_len], dtype=np.int32) + pad_ix
    positive = np.zeros([max_seq_len], dtype=np.int32) + pad_ix
    negative = np.zeros([max_seq_len], dtype=np.int32) + pad_ix

    # Sample negatives, moving backwards through the sequence
    user_excluded = set(user_sequences[user])
    ix = max_seq_len - 1
    next_item = user_sequences[user][-1]
    for i in reversed(user_sequences[user][:-1]):
        sequence[ix] = i
        positive[ix] = next_item
        if next_item != pad_ix:
            negative[ix] = random_sample_item(n_items, user_excluded)
        next_item = i
        ix -= 1
        if ix == -1:
            break

    return user, sequence, positive, negative

def sample_batch(batch_size, result_queue, seed, **sample_kwargs):
    """
    Sample batches

    Arguments:
        batch_size {int} -- Size of each batch
        result_queue {multiprocessing.Queue} -- Persistent queue for generated batches
        seed {int} -- Random seed for numpy
    """
    # Seed RNG
    np.random.seed(seed)

    # Continuously generate batches
    while True:
        # Sample a batch
        batch = [sample_sequence(**sample_kwargs) for _ in range(batch_size)]

        # Put the user, sequence, positives, negatives in queue
        result_queue.put(zip(*batch))

class WarpSampler(object):
    def __init__(self, user_sequences, n_users, n_items, pad_ix,
                 batch_size=64, max_seq_len=50, n_workers=1):
        """
        Sampler to randomly pick user sequences

        Arguments:
            user_sequences {map} -- user_sequences[u] is the interaction sequence for a user
            n_users {int} -- Number of unique users
            n_items {int} -- Number of unique items
            pad_ix {int} -- Index of padding item

        Keyword Arguments:
            batch_size {int} -- Size of each batch (default: {64})
            max_seq_len {int} -- Maximum length of an interaction sequence (default: {50})
            n_workers {int} -- Number of multiprocessing workers (default: {1})
        """
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_batch,
                    kwargs={
                        'batch_size': batch_size,
                        'result_queue': self.result_queue,
                        'seed': np.random.randint(2e9),
                        'user_sequences': user_sequences,
                        'n_users': n_users,
                        'n_items': n_items,
                        'max_seq_len': max_seq_len,
                        'pad_ix': pad_ix,
                    }
                ))

            # Set daemon = True to kill subprocesses after script ending
            # https://stackoverflow.com/questions/27494725/python-multiprocessing-daemon
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def load_sequences(loc, as_dict=True):
    """
    Load user sequence histories

    Arguments:
        loc {str} -- Sequence file location

    Keyword Arguments:
        as_dict {bool} -- Load sequences as a dictionary (default: {True})

    Returns:
        int -- Number of unique users
        int -- Number of unique items
        AND:
        3 x pd.DataFrame -- DataFrame of data (u, i)
            OR
        3 x dict -- Dictionary of user : list of items in order
    """
    # Load text
    if loc[-4:] == '.txt':
        df = pd.read_csv(
            loc, header=None, delim_whitespace=True,
            engine='c', names=['u', 'i']
        )
    # Load pickle
    else:
        df = pd.read_pickle(loc)

    # If not 0-indexed mappings
    df['u'] = df['u'] - df['u'].min()
    df['i'] = df['i'] - df['i'].min()
    n_users = df['u'].max()
    n_items = df['i'].max()

    # Log stats
    print_data_stats(df)

    # Get splits
    train, valid, test = get_splits(df, as_dict)

    return n_users, n_items, train, valid, test

def print_data_stats(df):
    n_users = df['u'].max()
    n_items = df['i'].max()
    n_ints = len(df)

    print('======= DF STATS =======')
    print('# Interactions: {:,}'.format(n_ints))
    print('# Users: {:,}'.format(n_users))
    print('# Items: {:,}'.format(n_items))
    print('Sparsity: {:,.3f}%'.format(
        100.0 * (1.0 - n_ints / (n_users * n_items))
    ))

def get_splits(df, as_dict=True):
    """
    Get training, test, and validation splits

    Arguments:
        df {pd.DataFrame} -- Interaction DF, sorted by user and interaction time

    Keyword Arguments:
        as_dict {bool} -- Load sequences as dictionaries (default: {True})
    """
    # Track # of interactions - if fewer than 3, use them for training only
    n_interactions = df.groupby('u').size()
    train_only = df[df['u'].isin(n_interactions[n_interactions < 3].index)]
    df_candidates = df.loc[~df.index.isin(train_only.index)]

    # Create holdout DFs
    holdout = df_candidates.groupby(['u'], as_index=False).tail(2)

    # Test - latest interaction
    holdout_test = holdout.groupby(['u'], as_index=False).tail(1)

    # Training - all interactions prior to last two
    train_df = df_candidates.loc[~df_candidates.index.isin(holdout.index)]
    train_df = pd.concat([train_df, train_only], axis=0, sort=False)

    # Validation - second-to-last interaction
    holdout_validation = holdout.loc[~holdout.index.isin(holdout_test.index)]

    # Return as dictionaries
    dfs = [train_df, holdout_validation, holdout_test]
    if as_dict:
        return [d.groupby('u')['i'].agg(list).to_dict() for d in dfs]

    return dfs

'''
python -m sasrec_torch.data --data data\\Beauty.txt
'''
if __name__ == "__main__":
    import os
    import argparse

    from datetime import datetime

    # Parse arguments
    parser = argparse.ArgumentParser(description='Testing samplers')
    parser.add_argument('--data', default=os.path.join('data', 'Beauty.txt'))
    args = parser.parse_args()

    # Read data
    data_loc = args.data
    start = datetime.now()
    n_users, n_items, train_seq, valid_seq, test_seq = load_sequences(data_loc, as_dict=True)
    pad_ix = n_items
    n_items_w_pad = n_items + 1
    print('{} - Loaded data for splits. Padding index: {}'.format(
        datetime.now() - start, pad_ix
    ))
