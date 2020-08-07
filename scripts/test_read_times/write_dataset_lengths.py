import os
import sys; sys.path[0] = '/export/home/tape/'
import argparse
import lmdb
import pickle
from functools import partial
from tqdm import tqdm
from tape.datasets import LMDBDataset
from multiprocessing import Pool

def create_parser():
    parser = argparse.ArgumentParser(description='Writes the lengths of a dataset to a file')
    parser.add_argument('--data_dir',
                        default='/export/home/tape/data/pfam/',
                        help='The dataset directory to read in',
                        type=str)
    parser.add_argument('--out_dir',
                        default='/export/home/tape/data/pfam/',
                        help='Where to write the list of lengths to',
                        type=str)
    parser.add_argument('--sequence_key',
                        default='primary',
                        help='Index of record to sequence',
                        type=str)
    parser.add_argument('--split',
                        choices=['all', 'holdout', 'valid', 'train'],
                        default='holdout',
                        help='The split to write')
    parser.add_argument('--num_jobs',
                        default=32,
                        help='The split to write',
                        type=int)
    return parser

def get_lengths_from_indices(indices, data_file, sequence_key):
    dataset = LMDBDataset(data_file)
    lengths = []
    for count, i in enumerate(indices):
        if count % int(len(indices) / 5) == 0:
            print(f'{i}/{indices[-1]}')
        length = len(dataset[i][sequence_key])
        lengths.append(length)
    return lengths

def write_split(split,
                data_dir,
                out_dir,
                sequence_key,
                num_jobs):
    data_file = os.path.join(data_dir, f'pfam_{split}.lmdb')
    dataset = LMDBDataset(data_file)

    apply_func = partial(get_lengths_from_indices, data_file=data_file, sequence_key=sequence_key)
    total_indices = list(range(len(dataset)))
    chunk_size = int(len(dataset) / num_jobs) + 1
    chunks = [list(range(i, min(i + chunk_size, len(dataset)))) for i in range(0, len(dataset), chunk_size)]

    out_file = os.path.join(out_dir, f'{split}_lengths.pkl')
    print(f'Writing lengths from {data_file} to {out_file}')
    with Pool(num_jobs) as pool:
        length_batches = pool.map(apply_func, chunks)
    sequence_lengths = [length for sublist in length_batches for length in sublist]
    with open(out_file, 'wb') as handle:
        pickle.dump(sequence_lengths, handle)

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    if args.split == 'all':
        for split in ['holdout', 'valid', 'train']:
            write_split(split=split,
                        data_dir=args.data_dir,
                        out_dir=args.out_dir,
                        sequence_key=args.sequence_key,
                        num_jobs=args.num_jobs)
    else:
        write_split(split=args.split,
                    data_dir=args.data_dir,
                    out_dir=args.out_dir,
                    sequence_key=args.sequence_key,
                    num_jobs=args.num_jobs)

if __name__ == '__main__':
    main()
