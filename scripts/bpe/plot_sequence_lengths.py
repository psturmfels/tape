import os
import sys; sys.path[0] = '/export/home/tape/'
import altair as alt
import numpy as np
import pandas as pd
import pickle
import argparse

from tqdm import tqdm
from tape.datasets import BPEMaskedLangaugeModelingDataset
from tape.tokenizers import BPETokenizer

def create_parser():
    parser = argparse.ArgumentParser(description='Computes tokenized vs. '
                                                 'untokenized protein sequence '
                                                 'lengths ')
    parser.add_argument('--split',
                        default='train',
                        choices=['train', 'valid', 'holdout'],
                        help='The split to compute lengths over.')
    parser.add_argument('--data_path',
                        default='/export/home/tape/data/naive_bpe',
                        type=str,
                        help='The directory containing the dataset.')
    parser.add_argument('--tokenizer_file',
                        default='/export/home/tape/scripts/bpe/naive_10k.pkl',
                        type=str,
                        help='A pickle file containing the tokenizer.')
    parser.add_argument('--output_dir',
                        default='/export/home/tape/scripts/bpe/',
                        type=str,
                        help='The directory to output charts to.')
    return parser

def read_lengths(data_path,
                 tokenizer,
                 split):
    dataset = BPEMaskedLangaugeModelingDataset(data_path=data_path,
                                               split=split,
                                               tokenizer=tokenizer)

    protein_lengths = np.zeros(len(dataset), dtype=int)
    token_lengths   = np.zeros(len(dataset), dtype=int)
    for i, item in enumerate(tqdm(dataset.data)):
        protein_lengths[i] = item['protein_length']
        token_lengths[i]   = len(item['primary'].split('$'))

    return protein_lengths, token_lengths

def aggregate_counts(counts, n_bins=10):
    min, max = np.min(counts), np.max(counts)
    q25, q75 = np.percentile(counts, [1, 99])
    bin_range = (q75 - q25) / n_bins

    bin_counts = dict([('{:.3f}-{:.3f}'.format(q25 + i * bin_range,
                                               q25 + (i + 1) * bin_range),
                        np.sum(np.logical_and(
                            counts > q25 + i * bin_range,
                            counts <= q25 + (i + 1) * bin_range
                        ))) for i in range(n_bins)])
    bin_counts['<{:.3f}'.format(q25)] = np.sum(counts <= q25)
    bin_counts['>{:.3f}'.format(q75)] = np.sum(counts > q75)
    ranges, values = list(zip(*bin_counts.items()))
    return pd.DataFrame({'Range': ranges,
                         'Value': values})

def plot_histogram(values, n_bins=10, color='darkblue', title=None):
    counts_df = aggregate_counts(values, n_bins=n_bins)
    print(counts_df)
    return alt.Chart(counts_df).mark_bar(color=color).encode(
        alt.X('Range:N', title='Length of Sequence', sort=None),
        alt.Y('Value:Q', title='Number of Sequences')
    ).properties(title=title)

def main(args=None):
    protein_lengths, token_lengths = read_lengths(data_path=args.data_path,
                                                  tokenizer=args.tokenizer_file,
                                                  split=args.split)
    print('Minimum protein length: {}, Maximum protein length: {}'.format(np.min(protein_lengths),
                                                                          np.max(protein_lengths)))
    print('Minimum protein length tokenized: {}, Maximum protein length tokenized: {}'.format(np.min(token_lengths),
                                                                                              np.max(token_lengths)))
    protein_histogram = plot_histogram(protein_lengths,
                                       color='darkblue',
                                       title='Length of Untokenized Protein Sequences')
    token_histogram   = plot_histogram(token_lengths,
                                       color='firebrick',
                                       title='Length of BPE-Tokenized Protein Sequences')

    protein_histogram.save(os.path.join(args.output_dir, 'protein_hist.png'))
    token_histogram.save(os.path.join(args.output_dir,   'token_hist.png'))

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
