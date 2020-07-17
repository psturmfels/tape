import os
import sys; sys.path[0] = '/export/home/tape/'
import argparse
import numpy as np

from tqdm import tqdm
from tape import utils
from tape.registry import registry

def create_parser():
    parser = argparse.ArgumentParser(description='Computes which tokens '
                                      ' are most mistaken for each other.')
    parser.add_argument('model_type', help='Base model class to run')
    parser.add_argument('--model_config_file', default=None, type=utils.check_is_file,
                        help='Config file for model')
    parser.add_argument('--output_file',
                        default='/export/home/tape/results/counts_matrix.npy',
                        type=str)
    parser.add_argument('--tokenizer', type=str, # choices=['iupac', 'unirep'],
                        default='iupac', help='Tokenizer to use on the amino acid sequences')
    parser.add_argument('--load_dir', default=None,
                        help='Directory containing config and pretrained model weights')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--data_dir', default='/export/home/tape/data/', type=utils.check_is_dir,
                        help='Directory from which to load task data')
    parser.add_argument('--split', default='holdout', type=str,
                        help='Which split to run on')
    return parser

def compute_batch_mismatch(batch_inputs, batch_outputs, tokenizer):
    """
    Computes the pairwise occurence counts between inputs and outputs.

    Args:
        batch_inputs: A dict containing a 'targets': tensor entry
        batch_outputs: A tuple (loss, softmax_predictions)
        tokenizer: A tokenizer object

    Returns:
        A len(tokens) x len(tokens) array corresponding to the mismatch counts
        of this batch.
    """
    batch_outputs = batch_outputs[1].detach().numpy()
    batch_outputs = np.argmax(batch_outputs, axis=-1).astype(int)
    batch_targets = batch_inputs['targets'].numpy().astype(int)

    mask_predict = batch_targets != -1
    true_ids = batch_targets[mask_predict]
    pred_ids = batch_outputs[mask_predict]

    hash_indices = np.maximum(true_ids, pred_ids) + \
                   tokenizer.vocab_size * np.minimum(true_ids, pred_ids)
    _, index_map, id_counts = np.unique(hash_indices,
                                        return_index=True,
                                        return_counts=True)

    mismatch_counts = np.zeros((tokenizer.vocab_size, tokenizer.vocab_size),
                               dtype=int)
    mismatch_counts[true_ids[index_map], pred_ids[index_map]] = id_counts
    mismatch_counts[pred_ids[index_map], true_ids[index_map]] = id_counts

    return mismatch_counts

def compute_matrix_mismatches(model, data_loader, tokenizer):
    num_batches = len(data_loader)
    counts_matrix = np.zeros((tokenizer.vocab_size, tokenizer.vocab_size),
                             dtype=int)
    for batch_inputs in tqdm(data_loader,
                             desc='Running Evaluation',
                             total=num_batches):
        batch_outputs = model(**batch_inputs)
        counts_matrix += compute_batch_mismatch(batch_inputs,
                                                batch_outputs,
                                                tokenizer)
    return counts_matrix

def main(args):
    model = registry.get_task_model(model_name=args.model_type,
                                    task_name='masked_language_modeling',
                                    config_file=args.model_config_file,
                                    load_dir=args.load_dir)
    valid_dataset = utils.setup_dataset(task='masked_language_modeling',
                                        data_dir=args.data_dir,
                                        split=args.split,
                                        tokenizer=args.tokenizer)
    valid_loader = utils.setup_loader(dataset=valid_dataset,
                                      batch_size=args.batch_size,
                                      local_rank=-1,
                                      n_gpu=1,
                                      gradient_accumulation_steps=1,
                                      num_workers=8)
    tokenizer = valid_dataset.tokenizer

    counts_matrix = compute_matrix_mismatches(model,
                                              valid_loader,
                                              tokenizer)
    np.save(args.output_file, counts_matrix)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
