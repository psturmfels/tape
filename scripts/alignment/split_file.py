import os
import subprocess
import argparse
from tqdm import tqdm

def create_parser():
    parser = argparse.ArgumentParser(description='Splits a file by a matched string')
    parser.add_argument('--input_file',
                        default='/export/home/tape/data/alignment/pfam/full/Pfam-A.full',
                        type=str,
                        help='A file to be split.')
    parser.add_argument('--output_directory',
                        default='/export/home/tape/data/alignment/pfam/aligned/',
                        type=str,
                        help='A directory to output split files')
    parser.add_argument('--split_string',
                        default='# STOCKHOLM 1.0',
                        type=str,
                        help='The string denoting the start of a block')
    parser.add_argument('--terminal_string',
                        default='//',
                        type=str,
                        help='The string denoting the end of a block')
    parser.add_argument('--name_string',
                        default='#=GF AC',
                        type=str,
                        help='The string that indicates a line containing accession id.')
    parser.add_argument('--extension',
                        default='aln',
                        choices=['aln', 'hmm'],
                        help='Output file type')
    parser.add_argument('--skip_to',
                        type=int,
                        default=None,
                        help='Skip a number of lines before starting')
    return parser

def write_lines_to_file(lines,
                        out_file):
    with open(out_file, 'w') as handle:
        for line in lines:
            handle.write(line)

def get_next_bounds(handle,
                    split_string,
                    terminal_string,
                    name_string):
    line = next(handle)
    lines = [line]
    if not line.startswith(split_string):
        raise StopIteration

    name = None
    count = 0
    while not line.startswith(terminal_string):
        if line.startswith(name_string):
            name = line.split()[-1]

        count += 1
        try:
            line = next(handle)
            lines.append(line)
        except UnicodeDecodeError as e:
            print(f'Found unicode decode error at count {count}')
            print(e)
            continue
    return count, name, lines

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    current_line_number = 1
    with open(args.input_file, 'r') as handle:
        if args.skip_to is not None:
            for i in tqdm(range(args.skip_to - 1)):
                _ = next(handle)
            current_line_number = args.skip_to

        while True:
            try:
                count, name, lines = get_next_bounds(handle,
                                                    args.split_string,
                                                    args.terminal_string,
                                                    args.name_string)
                print(name, current_line_number)
                output_file = os.path.join(args.output_directory,
                                           name + '.' + args.extension)
                write_lines_to_file(lines,
                                    output_file)
                current_line_number += count + 1
                del lines
            except StopIteration:
                break

if __name__ == '__main__':
    main()
