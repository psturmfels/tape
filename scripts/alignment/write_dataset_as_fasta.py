import os
import sys; sys.path[0] = '/export/home/tape'
import argparse

from Bio import SeqIO
from Bio import Alphabet
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from tqdm import tqdm
from tape.utils import setup_dataset
from tape.datasets import LMDBDataset

def create_parser():
    parser = argparse.ArgumentParser(description='Saves a dataset as a FASTA file')
    parser.add_argument('--task',
                        default='masked_language_modeling',
                        type=str,
                        help='The task to serialize')
    parser.add_argument('--split',
                        default='train',
                        choices=['train', 'valid', 'holdout'],
                        help='The split to tokenize.')
    parser.add_argument('--data_dir',
                        default='/export/home/tape/data/',
                        type=str,
                        help='The directory containing the original dataset.')
    parser.add_argument('--tokenizer',
                        default='iupac',
                        type=str,
                        help='A pickle file containing the tokenizer.')
    parser.add_argument('--output_file',
                        default='/export/home/tape/data/out.faa',
                        type=str,
                        help='The directory to output the fasta file to.')
    parser.add_argument('--restrict_id', action='store_true',
                        help='Set flag to restrict to a given family')
    parser.add_argument('--pfam_id',
                        default='PF03417.16',
                        type=str,
                        help='Required if --restrict_id is set. The id to restrict to.')
    parser.add_argument('--id_map_dir',
                        default='/export/home/tape/data/alignment/pfam/index_map/',
                        type=str,
                        help='Location of mapping from index to family.')
    return parser

def write_dataset_as_fasta(task,
                           split,
                           dataset,
                           output_file,
                           pfam_id=None,
                           family_dataset=None):
    sequence_records = []
    for item in tqdm(dataset.data, total=len(dataset)):
        sequence_length = item['protein_length']
        sequence_string = item['primary']
        sequence_family = item['family']
        sequence_clan   = item['clan']
        sequence_id     = item['id']

        sequence = Seq(sequence_string, alphabet=Alphabet.IUPAC)
        sequence_id_string = f'id: {sequence_id}|length: {sequence_length}|' + \
                             f'family: {sequence_family}|clan: {sequence_clan}|' + \
                             f'task: {task}|split: {split}'

        if family_dataset is not None:
            family_record = family_dataset[int(sequence_id)]
            sequence_pfam_id = family_record['pfam_id']
            sequence_species = family_record['species']
            sequence_uniprot_id  = family_record['uniprot_id']
            sequence_start_index = family_record['start_index']
            sequence_end_index   = family_record['end_index']
            sequence_id += f'|pfam_id: {sequence_pfam_id}|' + \
                           f'species: {sequence_species}|' + \
                           f'uniprot_id: {sequence_uniprot_id}|' + \
                           f'{sequence_start_index}-{sequence_end_index}|'

            if sequence_pfam_id != pfam_id:
                continue


        sequence_record = SeqRecord(seq=sequence,
                                    id=sequence_id_string,
                                    description=f'The {sequence_id}th sequence ' + \
                                                f'from {task} ({split})')
        sequence_records.append(sequence_record)

    with open(output_file, 'w') as handle:
        SeqIO.write(sequence_records, handle, 'fasta')

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    dataset = setup_dataset(task=args.task,
                            data_dir=args.data_dir,
                            split=args.split,
                            tokenizer=args.tokenizer)

    family_dataset = None
    if args.restrict_id:
        family_data_file = os.path.join(args.id_map_dir,
                                        f'pfam_{args.split}.lmdb')
        family_dataset = LMDBDataset(data_file=family_data_file)

    write_dataset_as_fasta(args.task,
                           args.split,
                           dataset,
                           args.output_file,
                           args.pfam_id,
                           family_dataset)

if __name__ == '__main__':
    main()
