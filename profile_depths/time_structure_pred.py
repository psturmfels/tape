import torch
import argparse
import os
import time
import sys
import numpy as np
import pandas as pd
from netsurfp2 import parse_fasta, preprocess, model, convert_npz

sys.path[0] = '/export/home/tape'
from tape import utils
from tape.registry import registry
from tape.training import ForwardRunner, run_eval_epoch

def create_parser():
    parser = argparse.ArgumentParser(description='Time models on secondary structure prediction')
    parser.add_argument('model_type',
                        choices=['netsurf', 'transformer'],
                        default='netsurf',
                        help='Model type to run')
    parser.add_argument('model_path',
                        type=str,
                        help='Where the model is located')
    parser.add_argument('out_file',
                        type=str,
                        help='Output file')
    parser.add_argument('split',
                        choices=['casp12', 'ts115', 'cb513'],
                        help='split to run on')

    parser.add_argument('--hhdb', choices=['scop70', 'uniclust30'],
                        help='HHBlits Database',
                        default='scop70')
    parser.add_argument('--n_threads', default=16, type=int, help='number of jobs')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=50)
    parser.add_argument('--no_cuda', help='Turn off gpus', action='store_true')

    return parser

def accuracy(labels, predictions):
    valid_mask = np.sum(labels, axis=-1).astype(int)
    sequence_lengths = np.sum(valid_mask, axis=-1)

    index_predictions = np.argmax(predictions, axis=-1)
    index_labels = np.argmax(labels, axis=-1)

    match = (index_predictions == index_labels).astype(int)
    index_sum = np.sum(valid_mask * match, axis=-1)

    accuracy = index_sum / sequence_lengths.astype(float)
    return accuracy

def run_netsurf(model_path, split, hhdb, n_threads, batch_size):
    fasta_file = f'/export/home/tape/profile_depths/{split}.faa'
    label_file = f'/export/home/tape/profile_depths/netsurfp2/{split.upper()}_HHblits.npz'
    tmp_out = f'/export/home/tape/profile_depths/{split}_{hhdb}/'

    print(f'Running netsurf on {fasta_file}, {hhdb} with {n_threads} threads and batch size of {batch_size}')
    with open(fasta_file) as fasta_handle:
        protlist = parse_fasta(fasta_handle)

    num_sequences = len(protlist)
    if hhdb == 'scop70':
        searcher = preprocess.HHblits('/export/home/tape/profile_depths/scop70/scop70_1.75', n_threads=n_threads)
    elif hhdb == 'uniclust30':
        searcher = preprocess.HHblits('/export/home/tape/profile_depths/uniclust30_2018_08/uniclust30_2018_08', n_threads=n_threads)
    computation_start = time.time()

    print('Building multiple sequence alignments...')
    search_start = time.time()
    profiles = searcher(protlist, tmp_out)
    search_elapsed = time.time() - search_start
    search_per_sequence = search_elapsed / num_sequences

    print('Running main netsurf model...')
    pred_start = time.time()
    nsp_model = model.TfGraphModel.load_graph(model_path)
    results = nsp_model.predict(profiles, tmp_out, batch_size=batch_size)
    pred_elapsed = time.time() - pred_start
    pred_per_sequence = pred_elapsed / num_sequences

    time_elapsed = time.time() - computation_start
    time_per_sequence = time_elapsed / num_sequences

    label_data = np.load(label_file)
    q8_labels = label_data['data'][:, :, 57:65]
    q3_labels = np.stack([np.sum(q8_labels[:, :, 0:3], axis=-1),
                          np.sum(q8_labels[:, :, 3:5], axis=-1),
                          np.sum(q8_labels[:, :, 5:8], axis=-1)],
                         axis=-1)
    ids = list(label_data['pdbids'])
    sorted_index = sorted(range(len(ids)), key = lambda k: ids[k])
    labels = np.stack([q3_labels[i] for i in sorted_index], axis=0)

    predictions = convert_npz(results)

    accuracy_per_sequence = accuracy(labels, predictions['q3'])
    acc = np.mean(accuracy_per_sequence)
    print(f'Mean accuracy: {acc:.4f}')
    print(f'Search time per sequence: {search_per_sequence:.2f}')
    print(f'Prediction time per sequence: {pred_per_sequence:.2f}')
    print(f'Total time per sequence: {time_per_sequence:.2f}')

    return acc, search_per_sequence, pred_per_sequence, time_per_sequence

def run_transformer(model_path, split, batch_size, no_cuda):
    local_rank = -1

    device, n_gpu, is_master = utils.setup_distributed(local_rank, no_cuda)
    model = registry.get_task_model('transformer', 'secondary_structure', None, model_path)
    model = model.to(device)

    runner = ForwardRunner(model, device, n_gpu)
    runner.initialize_distributed_model()

    valid_dataset = utils.setup_dataset('secondary_structure', '/export/home/tape/data/', split, 'iupac')
    num_sequences = len(valid_dataset)
    valid_loader = utils.setup_loader(valid_dataset, batch_size, local_rank, n_gpu, 1, 1)

    acc_fn = registry.get_metric('accuracy')
    computation_start = time.time()
    save_outputs = run_eval_epoch(valid_loader, runner, is_master)
    time_elapsed = time.time() - computation_start
    time_per_sequence = time_elapsed / num_sequences

    acc = acc_fn(save_outputs)

    print(f'Mean accuracy: {acc:.4f}')
    print(f'Search time per sequence: 0.00')
    print(f'Prediction time per sequence: {time_per_sequence:.2f}')
    print(f'Total time per sequence: {time_per_sequence:.2f}')

    return acc, 0.0, time_per_sequence, time_per_sequence

def save_to_csv(csv_file,
                model_type,
                accuracy,
                search_per_sequence,
                pred_per_sequence,
                time_per_sequence,
                split,
                hhdb,
                used_gpu):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        df = df.append({'model_type': model_type,
                        'accuracy': accuracy,
                        'search_per_sequence': search_per_sequence,
                        'pred_per_sequence': pred_per_sequence,
                        'time_per_sequence': time_per_sequence,
                        'split': split,
                        'hhdb': hhdb,
                        'used_gpu': used_gpu},
                       ignore_index=True)
    else:
        df = pd.DataFrame({'model_type': [model_type],
                           'accuracy': [accuracy],
                           'search_per_sequence': [search_per_sequence],
                           'pred_per_sequence': [pred_per_sequence],
                           'time_per_sequence': [time_per_sequence],
                           'split': [split],
                           'hhdb': [hhdb],
                           'used_gpu': [used_gpu]})
    df.to_csv(csv_file, index=False)

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    if args.model_type == 'netsurf':
        accuracy, search_per_sequence, pred_per_sequence, time_per_sequence = \
            run_netsurf(args.model_path,
                        args.split,
                        args.hhdb,
                        args.n_threads,
                        args.batch_size)
    else:
        accuracy, search_per_sequence, pred_per_sequence, time_per_sequence = \
            run_transformer(args.model_path,
                            args.split,
                            args.batch_size,
                            args.no_cuda)

    print(f'Saving results to {args.out_file}...')
    save_to_csv(args.out_file,
                args.model_type,
                accuracy,
                search_per_sequence,
                pred_per_sequence,
                time_per_sequence,
                args.split,
                args.hhdb,
                not args.no_cuda)

if __name__ == '__main__':
    main()
