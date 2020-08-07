import os
import sys; sys.path[0] = '/export/home/tape'
import argparse
import pickle
from tqdm import tqdm
from tape.utils.hmm import HMMReader, HMMContainer

def create_parser():
    parser = argparse.ArgumentParser(description='Reads in a bunch of hmms.')
    parser.add_argument('--data_dir',
                        default='/export/home/tape/data/alignment/pfam/hmm/',
                        help='Directory containing hmm files',
                        type=str)
    parser.add_argument('--output_file',
                        default='/export/home/tape/data/alignment/pfam/hmm/hmm_dict.pkl',
                        help='Where to output the dictionary',
                        type=str)
    return parser

def read_hmm_dictionary(data_dir):
    hmm_files = os.listdir(data_dir)
    hmm_files = [file for file in hmm_files if file.endswith('.hmm')]
    hmm_dictionary = {}
    for hmm_file in tqdm(hmm_files):
        hmm_header = os.path.splitext(hmm_file)[0]
        hmm_full_path = os.path.join(data_dir, hmm_file)
        try:
            with open(hmm_full_path, 'r') as handle:
                hmm_container = next(iter(HMMReader(handle)))
                hmm_dictionary[hmm_header] = hmm_container
        except StopIteration:
            continue
    return hmm_dictionary

def save_hmm_dictionary(hmm_dictionary, output_file):
    with open(output_file, 'wb') as handle:
        pickle.dump(hmm_dictionary, handle)

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    hmm_dictionary = read_hmm_dictionary(args.data_dir)
    save_hmm_dictionary(hmm_dictionary, args.output_file)

if __name__ == '__main__':
    main()
