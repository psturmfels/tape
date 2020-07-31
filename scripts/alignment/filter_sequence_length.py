import os
import sys; sys.path[0] = '/export/home/tape/'
import argparse
import lmdb
import pickle

from tqdm import tqdm
from tape.datasets import LMDBDataset

def create_parser():
    parser = argparse.ArgumentParser(description='Filters a dataset by sequence length')
    parser.add_argument('--input_file',
                        default='/export/home/tape/data/alignment/pfam/pfam_holdout.lmdb',
                        help='The dataset to filter',
                        type=str)
    parser.add_argument('--output_file',
                        default='/export/home/tape/data/alignment/pfam/pfam_holdout.lmdb',
                        help='The dataset to filter',
                        type=str)
    parser.add_argument('--length_key',
                        default='protein_length',
                        help='A key indicating either the length of the protein, ' + \
                              'or the protein sequence as a string.',
                        type=str)
    parser.add_argument('--maximum_length',
                        default=512,
                        help='Maximum length to filter out. Defaults to 512.',
                        type=int)
    parser.add_argument('--minimum_length',
                        default=16,
                        help='Minimum length to filter out. Defaults to 16.',
                        type=int)
    parser.add_argument('--map_size',
                        default=1e+13,
                        type=int,
                        help='LMDB write parameter.')
    return parser

def write_record_to_transaction(key, record, txn):
    _ = record.pop('id', None)
    txn.put(str(key).encode(), pickle.dumps(record))


def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    num_records = 0
    dataset = LMDBDataset(data_file=args.input_file)
    env = lmdb.open(args.output_file, map_size=args.map_size)
    with env.begin(write=True) as txn:
        for record in tqdm(dataset):
            if isinstance(record[args.length_key], str):
                length = len(record[args.length_key])
            elif isinstance(record[args.length_key], int):
                length = record[args.length_key]
            else:
                raise ValueError(f'Unrecognized length value {record[args.length_key]}')

            if args.maximum_length is not None and \
                length > args.maximum_length:
                print(record)
                continue
            if args.minimum_length is not None and \
                length < args.minimum_length:
                continue

            write_record_to_transaction(num_records,
                                        record,
                                        txn)
            num_records += 1
    print(f'Wrote {num_records}/{len(dataset)} records to {args.output_file}')

if __name__ == '__main__':
    main()
