import os
import argparse
import lmdb
import pickle
from tqdm import tqdm

import sys; sys.path[0] = '/export/home/tape/'
from tape.datasets import LMDBDataset

def create_parser():
    parser = argparse.ArgumentParser(description='Saves a mapping as an LMDB file')
    parser.add_argument('--input_directory',
                        default='/export/home/tape/data/alignment/pfam/index_map/',
                        type=str,
                        help='The directory containing the original dataset.')
    parser.add_argument('--output_directory',
                        default='/export/home/tape/data/alignment/pfam/index_map/',
                        type=str,
                        help='The directory to output the lmdb datasets to.')
    parser.add_argument('--write_reverse_index',
                        action='store_true',
                        help='Set to true to write the reverse index, '
                             'which maps from families to indices')
    parser.add_argument('--split',
                        choices=['train', 'valid', 'holdout', 'all'],
                        default='all',
                        help='Split to segment')
    return parser

def _add_value_to_key(txn,
                      key,
                      value):
    existing_record = txn.pop(key.encode())
    if existing_record is not None:
        existing_record = pickle.loads(existing_record)
        existing_record.extend(value)
        txn.put(key.encode(), pickle.dumps(existing_record))
    else:
        txn.put(key.encode(), pickle.dumps(value))

def _add_dict(txn,
              dict):
    for key, value in dict.items():
        _add_value_to_key(txn, key, value)

def write_reverse(input_directory,
                  output_directory,
                  map_size=1e+13,
                  split='train',
                  args=None,
                  batch_size=10000):
    data_file = os.path.join(input_directory, f'pfam_{split}.lmdb')
    dataset = LMDBDataset(data_file=data_file)

    print("Creating index map...")
    output_file = os.path.join(output_directory, f'reverse_{split}.lmdb')
    env = lmdb.open(output_file, map_size=map_size)
    with env.begin(write=True) as txn:
        batch_count = 0
        batch_dict = {}
        for item in tqdm(dataset, total=len(dataset)):
            dataset_index = int(item['dataset_index'])
            pfam_id = item['pfam_id']
            species = item['species']
            uniprot_id = item['uniprot_id']

            key = uniprot_id
            batch_count += 1
            if key in batch_dict:
                batch_dict[key].append(dataset_index)
            else:
                batch_dict[key] = [dataset_index]

            if batch_count >= batch_size:
                _add_dict(txn, batch_dict)
                batch_count = 0
                batch_dict = {}

def read_split(input_directory,
               output_directory,
               map_size=1e+11,
               split='train',
               args=None):
    output_file = os.path.join(output_directory, f'pfam_{split}.lmdb')
    env = lmdb.open(output_file, map_size=map_size)

    map_file = os.path.join(input_directory, f'pfam_{split}.map')
    num_examples = 0
    with open(map_file, 'r') as handle:
        with env.begin(write=True) as txn:
            for line in tqdm(handle):
                dataset_index, species, uniprot_id, pfam_id, start_index, end_index = \
                    line.split()

                data_dictionary = {
                    'dataset_index': int(dataset_index),
                    'species': species,
                    'uniprot_id': uniprot_id,
                    'pfam_id': pfam_id,
                    'start_index': int(start_index),
                    'end_index': int(end_index)
                }

                txn.put(dataset_index.encode(), pickle.dumps(data_dictionary))
                num_examples += 1
            txn.put('num_examples'.encode(), pickle.dumps(num_examples))

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    action_func = read_split
    if args.write_reverse_index:
        action_func = write_reverse

    if args.split == 'all':
        for split in ['holdout', 'valid', 'train']:
            print(f'Reading the {split} data...')
            action_func(input_directory=args.input_directory,
                        output_directory=args.output_directory,
                        split=split,
                        args=args)
    else:
        print(f'Reading the {args.split} data...')
        action_func(input_directory=args.input_directory,
                    output_directory=args.output_directory,
                    split=args.split,
                    args=args)

if __name__ == '__main__':
    main()
