'''
python -m sasrec_torch.model -d 3 -n 10 --data data\\Beauty.txt
'''
if __name__ == "__main__":
    import os
    import argparse

    from datetime import datetime

    from .model import SASRec
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
