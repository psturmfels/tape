import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

import sys; sys.path[0] = '/export/home/tape/'
from tape.utils import *


def create_parser():
    parser = argparse.ArgumentParser(description='plots relative accuracies')
    parser.add_argument('--split', default='casp12',
                        choices=['casp12', 'cb513', 'ts115'],
                        help='Which of the secondary structure test splits to use.')
    parser.add_argument('--results_dir',
                        default='/export/home/tape/results/secondary_structure_profile_prediction_transformer_full',
                        type=str,
                        help='Where the evaluated results are')
    parser.add_argument('--db', default='none',
                        choices=['scop40', 'scop70', 'uniclust30', 'none'],
                        help='Which background database')
    parser.add_argument('--sorted', action='store_true', help='Set to true to sort x axis')
    return parser

def accuracy_single(batch,
                    ignore_index=-1):
    pred  = batch['prediction']
    label = batch['target']

    pred_array  = pred.argmax(-1)
    mask = (label != ignore_index)
    is_correct = label[mask] == pred_array[mask]
    return is_correct.sum() / float(is_correct.size)

def accuracy(labels, predictions):
    valid_mask = np.sum(labels, axis=-1).astype(int)
    sequence_lengths = np.sum(valid_mask, axis=-1)

    index_predictions = np.argmax(predictions, axis=-1)
    index_labels = np.argmax(labels, axis=-1)

    match = (index_predictions == index_labels).astype(int)
    index_sum = np.sum(valid_mask * match, axis=-1)

    accuracy = index_sum / sequence_lengths.astype(float)
    return accuracy

def get_stats(hhm_file):
    with open(hhm_file, 'r') as handle:
        line = next(handle)
        while not line.strip().startswith('LENG'):
            line = next(handle)

        length = int(line.strip().split()[1])

        while not line.strip().startswith('FILT'):
            line = next(handle)
        num_seq = int(line.strip().split()[1])

        line = next(handle)
        neff = float(line.strip().split()[-1])

        while not line.strip().startswith('>Consensus'):
            line = next(handle)
        line = ''
        consensus = ''


        while not line.strip().startswith('>'):
            consensus += line.strip()
            line = next(handle)

        num_upper = np.sum([char.isupper() for char in consensus])
        num_total = np.sum([char != 'x' for char in consensus])
        num_lower = num_total - num_upper

        data = {
            'length': length,
            'num_seq': num_seq,
            'neff': neff,
            'upper': num_upper / length,
            'total': num_total / length,
            'lower': num_lower / length
        }

    return data

def get_neff(hhm_file):
    with open(hhm_file, 'r') as handle:
        line = next(handle)
        while not line.strip().startswith('NEFF'):
            line = next(handle)
    return float(line.strip().split()[-1])

def read_ids(fasta_file):
    ids = []
    with open(fasta_file, 'r') as h:
        for line in h:
            if line.startswith('>'):
                header = line.strip().split()[1]
                id = header.split('|')[0]
                ids.append(id)
    return ids

def read_stats(neffs_dir,
               ids):
    stats = []
    for file in ids:
        file_name = f'{file}/{file}_PROFILE.hhm'
        stat = get_stats(os.path.join(neffs_dir, file_name))
        stats.append(stat)
    return stats

def get_accuracies(transformer_results_file,
                   netsurf_results_file,
                   profile_file,
                   neffs_file,
                   ids_file,
                   split,
                   fasta_file):
    with open(transformer_results_file, 'rb') as handle:
        transformer_results = pickle.load(handle)
        transformer_results = transformer_results[1]
    dataset = setup_dataset(task='secondary_structure',
                            data_dir='/export/home/tape/data',
                            split=split)

    transformer_accuracies = [accuracy_single(batch) for batch in transformer_results]
    transformer_ids = [a['id'].decode() for a in dataset.data]
    transformer_dict = dict(zip(transformer_ids, transformer_accuracies))

    profiles  = np.load(profile_file)
    netsurf_ids = profiles['pdbids']

    if netsurf_results_file.endswith('.pkl'):
        with open(netsurf_results_file, 'rb') as handle:
            netsurf_results = pickle.load(handle)
        netsurf_labels = netsurf_results['labels']
        netsurf_predictions = netsurf_results['q3']
        netsurf_accuracies = accuracy(netsurf_labels, netsurf_predictions)
    else:
        reconstructed_netsurf_predictions = np.load(netsurf_results_file)['q3']
        reconstructed_netsurf_keys = list(sorted(list(set(netsurf_ids))))

        reconstructed_netsurf_dict = dict(zip(reconstructed_netsurf_keys, range(len(reconstructed_netsurf_keys))))
        reconstructed_netsurf_predictions = np.stack([reconstructed_netsurf_predictions[reconstructed_netsurf_dict[id]] for id in netsurf_ids], axis=0)

        q8_labels = profiles['data'][:, :, 57:65]
        q3_labels = np.stack([np.sum(q8_labels[:, :, 0:3], axis=-1),
                              np.sum(q8_labels[:, :, 3:5], axis=-1),
                              np.sum(q8_labels[:, :, 5:8], axis=-1)],
                             axis=-1)
        netsurf_labels = q3_labels

        netsurf_accuracies = accuracy(netsurf_labels, reconstructed_netsurf_predictions)

    netsurf_dict = dict(zip(netsurf_ids, netsurf_accuracies))

    if neffs_file.endswith('.npy'):
        file_names = np.load(ids_file)
        key_ids = [name.split('/')[-1].split('.hhr')[0] for name in file_names]
        neffs = np.load(neffs_file)
    else:
        stats = read_stats(neffs_file, netsurf_ids)

    neffs_dict = dict(zip(netsurf_ids, stats))

    accuracy_dict = {
        'id': [],
        'netsurf': [],
        'transformer': [],
        'neff': []
    }

    for id in neffs_dict.keys():
        netsurf_acc = netsurf_dict[id]
        transformer_acc = transformer_dict[id]
        neff = neffs_dict[id]

        accuracy_dict['id'].append(id)
        accuracy_dict['netsurf'].append(netsurf_acc)
        accuracy_dict['transformer'].append(transformer_acc)
        accuracy_dict['neff'].append(neff)

        # print(f'{id}:\t{netsurf_acc:.3f}\t{transformer_acc:.3f}')

    return accuracy_dict

def scatterplot_accuracy(accuracy_dict,
                         dpi=150,
                         title='Secondary Structure Accuracy',
                         sorted=False,
                         key='neff'):
    fig = plt.figure(dpi=dpi)
    ax = fig.gca()

    ax.grid()
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(0.1)
    ax.spines['right'].set_linewidth(0.1)
    ax.set_title(title, fontsize=18)

    fig.tight_layout()

    netsurf = np.array(accuracy_dict['netsurf'])
    transformer = np.array(accuracy_dict['transformer'])

    if sorted:
        diffs = transformer - netsurf
        indices = np.argsort(diffs)
        netsurf = netsurf[indices]
        transformer = transformer[indices]

        nonzero_index = np.where(diffs[indices] > 0)[0][0]
        x = np.arange(len(transformer))
        ax.axvline(nonzero_index, color='firebrick')
        ax.set_xlabel('Sorted Difference Index')
    else:
        x = [b[key] for b in accuracy_dict['neff']]
        ax.set_xlabel(key)

    ax.scatter(x,
               netsurf,
               c='tab:blue',
               label='Netsurfp2.0')
    ax.scatter(x,
               transformer,
               c='tab:orange',
               label='Profile Prediction')

    ax.set_ylabel('Q3 Accuracy')
    ax.legend()
    return fig, ax

def stratified_barplot(accuracy_dict):
    num_seqs = np.array([s['num_seq'] for s in accuracy_dict['neff']])
    transformer_accuracies = np.array(accuracy_dict['transformer'])
    netsurf_accuracies = np.array(accuracy_dict['netsurf'])

    buckets = [(1, 50), (50, 1000), (1000, 6000)]
    in_bucket = lambda neff, bucket: neff >= bucket[0] and neff < bucket[1]
    bucket_mask = lambda bucket: [in_bucket(ns, bucket) for ns in num_seqs]
    transformer_binned = [np.mean(transformer_accuracies[bucket_mask(bucket)]) for bucket in buckets]
    netsurf_binned = [np.mean(netsurf_accuracies[bucket_mask(bucket)]) for bucket in buckets]

    width = 0.4
    x_inds = np.arange(len(buckets))

    fig = plt.figure(dpi=150)
    ax = fig.gca()

    transformer_bar = ax.bar(x_inds - width/2, transformer_binned, width, label='Profile Prediction')
    netsurf_bar = ax.bar(x_inds + width/2, netsurf_binned, width, label='NetsurfP2.0')

    ax.set_ylabel('Secondary Structure Accuracy')
    ax.set_xlabel('Number of Related Sequences via MSA')
    ax.set_title('Accuracy Stratified by Alignment Depth', fontsize=14)
    ax.set_xticks(x_inds)
    ax.set_xticklabels(('<50 hits', '50-1000 hits', '>1000 hits'))
    ax.legend(loc='lower left')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['top'].set_linewidth(0.1)
    ax.spines['right'].set_linewidth(0.1)
    return fig, ax

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    base_path = '/export/home/tape/profile_depths/'

    profile_file = os.path.join(base_path, f'netsurfp2/{args.split.upper()}_HHblits.npz')
    if args.db == 'none':
        neffs_file = os.path.join(base_path, f'{args.split}/neffs.npy')
        netsurf_results_file = os.path.join(base_path, f'{args.split}_netsurf.pkl')
    else:
        neffs_file = os.path.join(base_path, f'{args.split}_{args.db}/')
        netsurf_results_file = os.path.join(base_path, f'{args.split}_{args.db}_netsurf.npz')

    fasta_file = os.path.join(base_path, f'{args.split}.faa')
    ids_file = os.path.join(base_path, f'{args.split}/result_files.npy')
    transformer_results_file = f'{args.results_dir}/{args.split}.pkl'

    print('Getting per-sequence accuracies...')
    accuracy_dict = get_accuracies(transformer_results_file,
                                   netsurf_results_file,
                                   profile_file,
                                   neffs_file,
                                   ids_file,
                                   args.split,
                                   fasta_file)

    os.makedirs(f'figures/{args.db}/', exist_ok=True)
    if args.sorted:
        output_file = os.path.join(base_path, f'figures/{args.db}/{args.split}_sorted.png')
        print(f'Writing figure to {output_file}')
        fig, ax = scatterplot_accuracy(accuracy_dict,
                                       title=f'Secondary Structure Accuracy ({args.split})',
                                       sorted=args.sorted,
                                       key=None)
        fig.savefig(output_file)
    else:
        for key in accuracy_dict['neff'][0].keys():

            output_file = os.path.join(base_path, f'figures/{args.db}/{args.split}_{key}.png')
            print(f'Writing figure to {output_file}')
            fig, ax = scatterplot_accuracy(accuracy_dict,
                                           title=f'Secondary Structure Accuracy ({args.split})',
                                           sorted=args.sorted,
                                           key=key)
            fig.savefig(output_file)

        output_file = os.path.join(base_path, f'figures/{args.db}/{args.split}_stratified_bar.png')
        print(f'Writing figure to {output_file}')
        fig, ax = stratified_barplot(accuracy_dict)
        fig.savefig(output_file)

if __name__ == '__main__':
    main()
