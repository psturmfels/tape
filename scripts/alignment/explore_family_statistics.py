import os
import sys; sys.path[0] = '/export/home/tape/'
import argparse
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm
from scipy.stats import describe

from hmm import HMMContainer, HMMReader
from tape.datasets import LMDBDataset

def create_parser():
    parser = argparse.ArgumentParser(description='Runs some statistical computations')
    parser.add_argument('method',
                        choices=['hmm', 'profiles'],
                        default='hmm')
    parser.add_argument('--split',
                        default='all',
                        choices=['train', 'valid', 'holdout', 'all'],
                        help='The split to compute counts over.')
    parser.add_argument('--data_dir',
                        default='/export/home/tape/data/alignment/pfam/index_map/',
                        type=str,
                        help='The directory containing the index mapping dataset.')
    parser.add_argument('--profile_dir',
                        default='/export/home/tape/data/alignment/pfam/hmm',
                        type=str,
                        help='The directory containing family profiles.')
    return parser

def compute_hmm_stats(data_dir='/export/home/tape/data/alignment/pfam/hmm/'):
    hmm_files = os.listdir(data_dir)
    hmm_files = list(filter(lambda file: file.endswith('.pkl'), hmm_files))

    lengths = []
    for file in tqdm(hmm_files):
        with open(os.path.join(data_dir, file), 'rb') as handle:
            hmm_container = pickle.load(handle)
            lengths.append(hmm_container.length)
    return np.array(lengths)

def get_dataset(data_dir, split):
    data_file = os.path.join(data_dir, f'pfam_{split}.lmdb')
    dataset = LMDBDataset(data_file=data_file)
    return dataset

def get_profile_list(profile_dir):
    profile_list = os.listdir(profile_dir)
    profile_list = [profile.split('.pkl')[0] for profile in profile_list if \
                    profile.endswith('.pkl')]
    profile_list = set(profile_list)
    return profile_list

def compute_missing_profiles(data_dir,
                             split,
                             profile_dir):
    print(f'Computing missing profiles for split {split}')
    dataset = get_dataset(data_dir, split)
    profile_list = get_profile_list(profile_dir)

    missing_profiles = {}
    for item in tqdm(dataset, total=len(dataset)):
        pfam_id = item['pfam_id']
        if pfam_id not in profile_list:
            missing_profiles[pfam_id] += 1

    return missing_profiles

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    if args.method == 'hmm':
        hmm_stats = compute_hmm_stats(args.data_dir)
        print(describe(hmm_stats))
    elif args.method == 'profiles':
        if args.split == 'all':
            for split in ['holdout', 'valid', 'train']:
                missing_profiles = compute_missing_profiles(args.data_dir,
                                                            split,
                                                            args.profile_dir)
                print(f'Split: {split}')
                print(missing_profiles)
        else:
            missing_profiles = compute_missing_profiles(args.data_dir,
                                                        args.split,
                                                        args.profile_dir)
            print(f'Split: {args.split}')
            print(missing_profiles)

if __name__ == '__main__':
    main()
