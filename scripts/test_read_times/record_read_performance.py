import os
import sys; sys.path[0] = '/export/home/tape'
import argparse
import time
from tqdm import tqdm
from tape.utils import setup_dataset, setup_loader

def create_parser():
    parser = argparse.ArgumentParser(description='Functionality for testing read speed.')
    parser.add_argument('--task', default='masked_language_modeling',
                        help='Tape task to benchmark', type=str)
    parser.add_argument('--split', default='holdout',
                        choices=['train', 'valid', 'holdout'])
    parser.add_argument('--data_dir', default='/export/home/tape/data/',
                        help='Location of data', type=str)
    parser.add_argument('--tokenizer', default='iupac',
                        help='Tokenizer to use', type=str)
    parser.add_argument('--batch_size', default=None,
                        help='Set to an int to use batching', type=int)
    parser.add_argument('--num_workers', default=None,
                        help='Only used in batch mode', type=int)
    parser.add_argument('--num_total', default=10000,
                        help='Number of reads to perform', type=int)
    parser.add_argument('--precomputed_key_file',
                        default=None,
                        help='Where the lengths of the records are stored',
                        type=str)
    return parser

def get_data_iter(task,
                  data_dir,
                  split,
                  tokenizer,
                  batch_size,
                  num_workers,
                  precomputed_key_file):
    dataset = setup_dataset(task=task,
                            data_dir=data_dir,
                            split=split,
                            tokenizer=tokenizer)
    if batch_size is not None:
        data_loader = setup_loader(dataset=dataset,
                                   batch_size=batch_size,
                                   local_rank=-1,
                                   n_gpu=1,
                                   gradient_accumulation_steps=1,
                                   num_workers=num_workers,
                                   precomputed_key_file=precomputed_key_file)
        return data_loader
    else:
        return dataset

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    data_iter = get_data_iter(task=args.task,
                              data_dir=args.data_dir,
                              split=args.split,
                              tokenizer=args.tokenizer,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              precomputed_key_file=args.precomputed_key_file)
    start = time.time()
    if args.batch_size is not None:
        total_iters = int((args.num_total / args.batch_size) + 0.5)
    else:
        total_iters = args.num_total

    for i, item in enumerate(tqdm(data_iter, total=total_iters)):
        if i >= total_iters:
            break

    end = time.time()
    print(f'For {args.num_total} iterations, time elapsed: {end-start:.5f} seconds')

if __name__ == '__main__':
    main()
