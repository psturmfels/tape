import os
import sys; sys.path.append('/export/home/tape/scripts/alignment')
import argparse
import numpy as np
import lmdb
import pickle
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from hmm import HMMContainer, HMMReader

from tape.datasets import LMDBDataset
from tape.utils import setup_dataset

LABEL_VOCAB = ['A',
               'C',
               'D',
               'E',
               'F',
               'G',
               'H',
               'I',
               'K',
               'L',
               'M',
               'N',
               'P',
               'Q',
               'R',
               'S',
               'T',
               'V',
               'W',
               'Y']

def create_parser():
    parser = argparse.ArgumentParser(description='Writes the dataset with HMM profile labels.')
    parser.add_argument('--input_directory',
                        default='/export/home/tape/data/alignment/pfam/test/',
                        type=str,
                        help='A folder containing aligned and hmm sub-folders')
    parser.add_argument('--output_directory',
                        default='/export/home/tape/data/alignment/pfam/test/',
                        type=str,
                        help='The directory to output the lmdb datasets to.')
    parser.add_argument('--num_processes',
                        default=100,
                        type=int,
                        help='Number of processes to run')
    parser.add_argument('--batch_size',
                        default=100,
                        type=int,
                        help='Number of families to process before writing.')
    parser.add_argument('--map_size',
                        default=1e+13,
                        type=int,
                        help='LMDB write parameter.')
    return parser

def read_alignment_file(file):
    uniprot_ids = []
    species_list = []
    sequence_ranges = []
    sequences = []
    reference_line = None
    with open(file, 'r') as handle:
        for line in handle:
            if line.startswith('#=GC RF'):
                reference_line = line.strip().split()[-1]
            elif not line.startswith('#') and not line.startswith('//'):
                split_line = line.strip().split()
                header = split_line[0]
                group, seq_range = header.split('/')
                uniprot_id, species = group.split('_')
                seq_range = tuple(int(r) for r in seq_range.split('-'))
                sequence = split_line[1]

                uniprot_ids.append(uniprot_id)
                species_list.append(species)
                sequence_ranges.append(seq_range)
                sequences.append(sequence)
    return uniprot_ids, species_list, sequence_ranges, sequences, reference_line

def read_hmm_file(file):
    with open(file, 'r') as handle:
        reader = HMMReader(handle)
        container = next(reader)
    return container

def get_label_from_hmm(reference_line, sequence, hmm_container):
    current_match_index = 0
    labels = []
    stripped_sequence = ''

    for index, character in enumerate(sequence):
        if character == '-':
            current_match_index += 1
        elif reference_line[index] == 'M':
            # Note: map index is 1-indexed, not 0-indexed
            assert index == hmm_container.map_index[current_match_index] - 1
            labels.append(hmm_container.probabilities['match'][current_match_index])
            current_match_index += 1
            stripped_sequence += character.upper()
        elif character.islower():
            stripped_sequence += character.upper()
            if current_match_index == 0:
                labels.append(hmm_container.probabilities['compo_insertion'])
            else:
                labels.append(hmm_container.probabilities['insertion'][current_match_index - 1])
        else:
            continue

    labels = np.array(labels)
    return labels, stripped_sequence

def write_sequence(txn,
                   key,
                   dataset_index,
                   reference_line,
                   raw_sequence,
                   hmm_container):
    label, sequence = get_label_from_hmm(reference_line, raw_sequence, hmm_container)
    data = {
        'primary': sequence,
        'profile': label,
        'protein_length': len(sequence),
        'dataset_index': int(dataset_index)
    }
    txn.put(str(key).encode(), pickle.dumps(data))

def batch_write_sequences(txn_dict,
                          count_dict,
                          batch_sequences_to_write,
                          split_list,
                          index_list,
                          hmm_container,
                          reference_line):
    if len(batch_sequences_to_write) == 0:
        return

    local_counts = {
        'train': 0,
        'valid': 0,
        'holdout': 0
    }
    for split, dataset_index, sequence in zip(split_list,
                                              index_list,
                                              batch_sequences_to_write):
        write_sequence(txn_dict[split],
                       count_dict[split] + local_counts[split],
                       dataset_index,
                       reference_line,
                       sequence,
                       hmm_container)
        local_counts[split] += 1

def cross_reference_indices(mapping,
                            forward,
                            uniprot_id,
                            sequence_range,
                            tolerance=3):
    indices = mapping.get(uniprot_id.encode())
    if indices is not None:
        indices = pickle.loads(indices)
        for index in indices:
            data_dictionary = forward[index]
            start_index = data_dictionary['start_index']
            end_index   = data_dictionary['end_index']
            if abs(start_index - sequence_range[0]) < tolerance and \
                abs(end_index - sequence_range[1]) < tolerance:
                return index
    return None

def cross_reference_tape(train_mapping,
                         valid_mapping,
                         holdout_mapping,
                         train_forward,
                         valid_forward,
                         holdout_forward,
                         uniprot_id,
                         sequence_range,
                         tolerance=3):
    train_index = cross_reference_indices(train_mapping,
                                          train_forward,
                                          uniprot_id,
                                          sequence_range,
                                          tolerance=tolerance)
    if train_index is not None:
        return 'train', train_index

    valid_index = cross_reference_indices(valid_mapping,
                                          valid_forward,
                                          uniprot_id,
                                          sequence_range,
                                          tolerance=tolerance)
    if valid_index is not None:
        return 'valid', valid_index

    holdout_index = cross_reference_indices(holdout_mapping,
                                            holdout_forward,
                                            uniprot_id,
                                            sequence_range,
                                            tolerance=tolerance)
    if holdout_index is not None:
        return 'holdout', holdout_index
    return None, None

def get_batch_write_from_alignment(train_mapping,
                                   valid_mapping,
                                   holdout_mapping,
                                   train_forward,
                                   valid_forward,
                                   holdout_forward,
                                   hmm_file,
                                   alignment_file):
    uniprot_ids, species_list, sequence_ranges, sequences, reference_line = read_alignment_file(alignment_file)
    hmm_container = read_hmm_file(hmm_file)
    assert hmm_container.outputs == LABEL_VOCAB


    batch_sequences_to_write = []
    split_list = []
    index_list = []

    for index, sequence in enumerate(sequences):
        uniprot_id = uniprot_ids[index]
        sequence_range = sequence_ranges[index]
        split, dataset_index = cross_reference_tape(train_mapping,
                                                    valid_mapping,
                                                    holdout_mapping,
                                                    train_forward,
                                                    valid_forward,
                                                    holdout_forward,
                                                    uniprot_id,
                                                    sequence_range)
        if split is not None:
            batch_sequences_to_write.append(sequence)
            split_list.append(split)
            index_list.append(dataset_index)
    return batch_sequences_to_write, split_list, index_list, hmm_container, reference_line

def get_batch_write_from_family(family,
                                base_dir):
    train_mapping, valid_mapping, holdout_mapping, train_env, valid_env, holdout_env = read_mappings()
    train_forward, valid_forward, holdout_forward = read_forwards()
    alignment_file = os.path.join(base_dir, 'aligned/', family + '.aln')
    hmm_file = os.path.join(base_dir, 'hmm/', family + '.hmm')
    result =  get_batch_write_from_alignment(train_mapping,
                                             valid_mapping,
                                             holdout_mapping,
                                             train_forward,
                                             valid_forward,
                                             holdout_forward,
                                             hmm_file,
                                             alignment_file)
    train_env.close()
    valid_env.close()
    holdout_env.close()
    return result

def read_mappings():
    train_env = lmdb.open('/export/home/tape/data/alignment/pfam/index_map/reverse_train.lmdb')
    train_mapping = train_env.begin(write=False)

    valid_env = lmdb.open('/export/home/tape/data/alignment/pfam/index_map/reverse_valid.lmdb')
    valid_mapping = valid_env.begin(write=False)

    holdout_env = lmdb.open('/export/home/tape/data/alignment/pfam/index_map/reverse_holdout.lmdb')
    holdout_mapping = holdout_env.begin(write=False)
    return train_mapping, valid_mapping, holdout_mapping, train_env, valid_env, holdout_env

def read_forwards():
    train_forward = LMDBDataset(data_file='/export/home/tape/data/alignment/pfam/index_map/pfam_train.lmdb')
    valid_forward = LMDBDataset(data_file='/export/home/tape/data/alignment/pfam/index_map/pfam_valid.lmdb')
    holdout_forward = LMDBDataset(data_file='/export/home/tape/data/alignment/pfam/index_map/pfam_holdout.lmdb')
    return train_forward, valid_forward, holdout_forward

def read_datasets():
    train_dataset = setup_dataset(task='masked_language_modeling',
                                  data_dir='/export/home/tape/data/',
                                  split='train',
                                  tokenizer='iupac')
    valid_dataset = setup_dataset(task='masked_language_modeling',
                                  data_dir='/export/home/tape/data/',
                                  split='valid',
                                  tokenizer='iupac')
    holdout_dataset = setup_dataset(task='masked_language_modeling',
                                    data_dir='/export/home/tape/data/',
                                    split='holdout',
                                    tokenizer='iupac')
    return train_dataset, valid_dataset, holdout_dataset

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    train_env = lmdb.open(os.path.join(args.output_directory, 'pfam_train.lmdb'), map_size=args.map_size)
    valid_env = lmdb.open(os.path.join(args.output_directory, 'pfam_valid.lmdb'), map_size=args.map_size)
    holdout_env = lmdb.open(os.path.join(args.output_directory, 'pfam_holdout.lmdb'), map_size=args.map_size)

    families = os.listdir(os.path.join(args.input_directory, 'aligned/'))
    families = ['.'.join(fam.split('.')[:-1]) for fam in families if fam.endswith('.aln')]
    print(f'Found {len(families)} to write...')

    apply_func = partial(get_batch_write_from_family,
                         base_dir=args.input_directory)

    print('Beginning write environments...')
    with train_env.begin(write=True) as train_txn:
        with valid_env.begin(write=True) as valid_txn:
            with holdout_env.begin(write=True) as holdout_txn:
                txn_dict = {
                    'train': train_txn,
                    'valid': valid_txn,
                    'holdout': holdout_txn
                }
                count_dict = {
                    'train': 0,
                    'valid': 0,
                    'holdout': 0
                }
                for i in range(0, len(families), args.batch_size):
                    print(f'-----Index {i}/{len(families)}-----')
                    family_batch = families[i:min(i + args.batch_size, len(families))]
                    with Pool(processes=args.num_processes) as pool:
                        batch_writes = pool.map(apply_func, family_batch)

                    for batch_sequences_to_write, split_list, index_list, \
                        hmm_container, reference_line in batch_writes:
                        print(f'Writing {len(batch_sequences_to_write)} from family {hmm_container.accession}...')
                        batch_write_sequences(txn_dict,
                                              count_dict,
                                              batch_sequences_to_write,
                                              split_list,
                                              index_list,
                                              hmm_container,
                                              reference_line)
                        for split in ['train', 'valid', 'holdout']:
                            count_dict[split] += split_list.count(split)


                for split in ['train', 'valid', 'holdout']:
                    txn_dict[split].put('num_examples'.encode(),
                                        pickle.dumps(count_dict[split]))

    print('Finished!')

if __name__ == '__main__':
    main()
