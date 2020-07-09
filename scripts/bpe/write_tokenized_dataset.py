import os
import sys
sys.path[0] = '/export/home/tape/'

import numpy as np
import torch
import tape
import pickle
import multiprocessing
import lmdb
import argparse
from tape import utils
from tape.tokenizers import TAPETokenizer, BPETokenizer
from tqdm import tqdm

def create_parser():
    parser = argparse.ArgumentParser(description='Saves a tokenized dataset in lmdb format')
    parser.add_argument('--split',
                        default='train',
                        choices=['train', 'valid', 'holdout'],
                        help='The split to tokenize.')
    parser.add_argument('--input_data_directory',
                        default='/export/home/tape/data/',
                        type=str,
                        help='The directory containing the original dataset.')
    parser.add_argument('--tokenizer_file',
                        default='/export/home/tape/scripts/bpe/naive_10k.pkl',
                        type=str,
                        help='A pickle file containing the tokenizer.')
    parser.add_argument('--output_directory',
                        default='/export/home/tape/data/naive_bpe/',
                        type=str,
                        help='The directory to output the tokenized dataset to.')
    parser.add_argument('--num_processes',
                        default=100,
                        type=int,
                        help='The number of simultaneous processes to run.')
    parser.add_argument('--block_size',
                        default=5000,
                        type=int,
                        help='The number of sequences to simultaneously keep '
                             'in memory.')
    parser.add_argument('--map_size',
                        default=1e11,
                        type=int,
                        help='Passed to lmdb dataset. Reserved file size.')
    return parser

def tokenize_indices(dataset, indices, tokenizer, q):
    tokenized_data = []
    for index in indices:
        item = dataset.data[index]
        sequence = item['primary']
        tokenized_sequence = tokenizer.tokenize(sequence)
        tokenized_sequence = '$'.join(tokenized_sequence)

        tokenized_item = {
            'primary': tokenized_sequence,
            'num_tokens': len(tokenized_sequence),
            'protein_length': item['protein_length'],
            'clan': item['clan'],
            'family': item['family'],
            'id': item['id']
        }
        tokenized_data.append(tokenized_item)
    q.put(tokenized_data)

def tokenize_dataset(dataset,
                     tokenizer,
                     env,
                     num_processes=10,
                     block_size=100000):
    proc_increment = int(block_size / num_processes + 0.5)

    with env.begin(write=True) as txn:
        txn.put('num_examples'.encode(), pickle.dumps(len(dataset)))
        for start_index in tqdm(range(0, len(dataset), block_size)):
            q = multiprocessing.Queue()
            proc_list = []
            for proc in range(num_processes):
                lower_index = start_index + proc_increment * proc
                upper_index = min(len(dataset),
                                  start_index + proc_increment * (proc + 1))
                proc_indices = list(range(lower_index, upper_index))
                process = multiprocessing.Process(target=tokenize_indices,
                                                   args=(dataset,
                                                         proc_indices,
                                                         tokenizer,
                                                         q))
                process.start()
                proc_list.append(process)
            for p in proc_list:
                tokenized_block = q.get()
                for sequence_dict in tokenized_block:
                    txn.put(str(sequence_dict['id']).encode(), pickle.dumps(sequence_dict))
            for p in proc_list:
                p.join()


def main(args=None):
    dataset = tape.datasets.MaskedLanguageModelingDataset(data_path=args.input_data_directory,
                                                          split=args.split)

    with open(args.tokenizer_file, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    output_file = os.path.join(args.output_directory,
                               f'pfam/pfam_{args.split}.lmdb')
    env = lmdb.open(output_file, map_size=1e11)
    tokenize_dataset(dataset=dataset,
                     tokenizer=tokenizer,
                     env=env,
                     num_processes=args.num_processes,
                     block_size=args.block_size)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
