import os
import sys
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

def create_parser():
    parser = argparse.ArgumentParser(description='Builds hmms from alignments.')
    parser.add_argument('--input_directory',
                        default='/export/home/tape/data/alignment/pfam/aligned/',
                        help='Directory containing files',
                        type=str)
    parser.add_argument('--extension',
                        default='aln',
                        help='The extension of the aligned files.',
                        type=str)
    parser.add_argument('--output_directory',
                        default='/export/home/tape/data/alignment/pfam/hmm/',
                        help='Where to put the hmm files',
                        type=str)
    parser.add_argument('--reference_number',
                        default=10,
                        help='Number of sequences to take reference over.',
                        type=int)
    parser.add_argument('--processes',
                        default=10,
                        help='Number of processes to run.',
                        type=int)
    parser.add_argument('--batch_size',
                        default=100,
                        help='Size of batch.',
                        type=int)
    return parser


def convert_sequences_to_annotation(sequences):
    reference = ''
    for index in range(len(sequences[0])):
        count_match = 0
        count_indel = 0
        for sequence in sequences:
            if sequence[index] == '.' or \
                sequence[index].islower():
                count_indel += 1
            elif sequence[index] == '-' or \
                sequence[index].isupper():
                count_match += 1
            else:
                raise ValueError(f'Unrecognized character {sequence[index]} ' +
                                 f'in sequence {sequence}')
        if count_match > count_indel:
            reference += 'M'
        else:
            reference += '.'
    return reference

def convert_reference_to_line(sequence_line, reference):
    begin_string = '#=GC RF'
    first_non_space = sequence_line.strip().split()[-1]
    header_size     = sequence_line.index(first_non_space)
    number_of_spaces = header_size - len(begin_string)
    reference_line = begin_string + ' ' * number_of_spaces + reference + '\n'
    return reference_line

def add_reference_annotation_to_file(file,
                                     reference_number=10):
    lines = []
    seq_start_index = None
    seq_end_index   = None
    with open(file, 'r') as handle:
        line_count = 0
        while True:
            try:
                line = next(handle)
            except StopIteration:
                break

            if line.startswith('#=GC RF'):
                return

            if not line.startswith('#') and not line.startswith('//'):
                if seq_start_index is None:
                    seq_start_index = line_count
            else:
                if seq_start_index is not None and seq_end_index is None:
                    seq_end_index = line_count

            lines.append(line)
            line_count += 1
    if seq_start_index is None:
        return

    seq_end_index = min(seq_start_index + 10, seq_end_index)
    sequences = [line.strip().split()[-1] \
                for line in lines[seq_start_index:seq_end_index]]
    reference_annotation = convert_sequences_to_annotation(sequences)
    reference_line = convert_reference_to_line(lines[seq_start_index], reference_annotation)
    lines.insert(seq_start_index, reference_line)

    with open(file, 'w') as handle:
        for line in lines:
            handle.write(line)

def build_hmm_from_file(input_file,
                        output_file):
    os.system(f'hmmbuild --hand {output_file} {input_file} > /dev/null')

def process_file(file,
                 reference_number,
                 output_directory):
    add_reference_annotation_to_file(file,
                                     reference_number=reference_number)
    file_base = os.path.basename(file)
    file_base = '.'.join(file_base.split('.')[:-1])
    file_base += '.hmm'
    output_file = os.path.join(output_directory, file_base)
    if not os.path.exists(output_file):
        build_hmm_from_file(file, output_file)

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    files = os.listdir(args.input_directory)
    files = [file for file in files if file.endswith(f'.{args.extension}')]
    files = [os.path.join(args.input_directory, file) for file in files]

    with Pool(processes=args.processes) as pool:
        _ = list(tqdm(pool.imap(partial(process_file,
                                        reference_number=args.reference_number,
                                        output_directory=args.output_directory),
                                files), total=len(files)))

if __name__ == '__main__':
    main()
