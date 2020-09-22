import sys; sys.path[0] = '/export/home/tape/profile_depths/netsurfp2/'
import os
import numpy as np
import model
import pickle
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Runs the netsurf model')
    parser.add_argument('--split', default='casp12',
                        choices=['casp12', 'ts115', 'cb513'],
                        help='Test split to run on.')
    parser.add_argument('--model_file',
                        default='/export/home/tape/profile_depths/netsurfp2/models/hhsuite.pb',
                        type=str,
                        help='File of model graph.')
    return parser

def save_predictions_and_labels(profile_file,
                                model_file,
                                output_file):
    nsp_model = model.TfGraphModel.load_graph(model_file)
    profiles  = np.load(profile_file)
    profile_array = profiles['data'][:, :, 0:51]
    predictions = nsp_model._predict_array(profile_array)

    q8_labels = profiles['data'][:, :, 57:65]
    q3_labels = np.stack([np.sum(q8_labels[:, :, 0:3], axis=-1),
                          np.sum(q8_labels[:, :, 3:5], axis=-1),
                          np.sum(q8_labels[:, :, 5:8], axis=-1)],
                         axis=-1)

    np.sum(q3_labels, axis=(1,2))
    predictions['labels'] = q3_labels
    with open(output_file, 'wb') as handle:
        pickle.dump(predictions, handle)

    return predictions, profiles

def accuracy(labels, predictions):
    valid_mask = np.sum(labels, axis=-1).astype(int)
    sequence_lengths = np.sum(valid_mask, axis=-1)

    index_predictions = np.argmax(predictions, axis=-1)
    index_labels = np.argmax(labels, axis=-1)

    match = (index_predictions == index_labels).astype(int)
    index_sum = np.sum(valid_mask * match, axis=-1)

    accuracy = index_sum / sequence_lengths.astype(float)
    return accuracy

def main(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    profile_file = f'/export/home/tape/profile_depths/netsurfp2/{args.split.upper()}_HHblits.npz'
    model_file = args.model_file
    output_file = f'/export/home/tape/profile_depths/netsurfp2/{args.split}_netsurf.pkl'

    predictions, profiles = save_predictions_and_labels(profile_file,
                                                        model_file,
                                                        output_file)

    accuracies = accuracy(predictions['labels'], predictions['q3'])
    ids = profiles['pdbids']

    file_names = np.load(f'/export/home/tape/profile_depths/{args.split}/result_files.npy')
    key_ids = [name.split('/')[-1].split('.hhr')[0] for name in file_names]
    neffs = np.load(f'/export/home/tape/profile_depths/{args.split}/neffs.npy')
    neffs_dict = dict(zip(key_ids, neffs))

    for id, acc in zip(ids, accuracies):
        neff = neffs_dict.get(id, None)
        print(f'{id}:\t{acc:.3f}\t{neff:.3f}')

if __name__ == '__main__':
    main()
