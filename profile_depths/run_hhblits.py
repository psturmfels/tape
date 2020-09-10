import os
import sys
import argparse
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser(description='Run hhblits on fasta files')
    parser.add_argument('--input_file', default='secondary_structure.faa',
                        type=str, help='Input fasta file')
    parser.add_argument('--out_dir', default='secondary_structure/',
                        type=str, help='Temporary single line fasta file directory')
    parser.add_argument('--database_path', default='scop40/scop40',
                        type=str, help='Location of reference database')
    return parser

def split_fasta(input_file,
                out_dir):
    out_files = []
    ids = []
    os.makedirs(out_dir, exist_ok=True)
    with open(input_file, 'r') as handle:
        fragment = next(handle)
        run = True
        while run:
            header = fragment
            sequence = ''
            while True:
                try:
                    fragment = next(handle).strip()
                except StopIteration:
                    run = False
                    break

                if fragment.startswith('>id:'):
                    break
                sequence += fragment

            id = header.split('|')[0].split(' ')[-1]
            out_file = os.path.join(out_dir, id + '.faa')
            out_files.append(out_file)
            ids.append(id)
            print(f'Writing to {out_file}')

            with open(out_file, 'w') as write_handle:
                write_handle.write(f'{header}\n')
                write_handle.write(f'{sequence}\n')
    return out_files, ids

def run_hhblits(out_dir, out_files, ids, database_path):
    result_files = []
    neffs = []
    for out_file, id in zip(out_files, ids):
        result_file = os.path.join(out_dir, id + '.hhr')
        command = f"hhblits -i {out_file} -o {result_file} -d {database_path}"
        print(f"======={command}=======")
        os.system(command)
        result_files.append(result_file)
        neffs.append(get_neff(result_file))
    return result_files, neffs

def get_neff(result_file):
    with open(result_file, 'r') as handle:
        for _ in range(4):
            line = next(handle)
        neff = float(line.split()[-1])
    return neff

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()
    out_files, ids = split_fasta(args.input_file, args.out_dir)
    result_files, neffs = run_hhblits(args.out_dir, out_files, ids, args.database_path)
    result_files = np.array(result_files)
    neffs = np.array(neffs)
    np.save(os.path.join(args.out_dir, 'result_files.npy'), result_files)
    np.save(os.path.join(args.out_dir, 'neffs.npy'), neffs)
    for r, n in zip(result_files, neffs):
        print(f'{r}:\t{n}')

if __name__ == '__main__':
    main()
