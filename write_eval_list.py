#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
import re
import sys
import argparse
import shlex
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tape.main import *
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Evaluates tape models')
    parser.add_argument('--split',
                        choices=['valid', 'test'],
                        default='valid',
                        help='Split to evaluate on')
    parser.add_argument('--results_dir',
                        default='/export/home/tape/results/',
                        type=str,
                        help='Where results are located')
    parser.add_argument('--out_file',
                        default='/export/home/tape/eval_results.csv',
                        type=str,
                        help='csv file to output')
    return parser

TASKS = ['secondary_structure',
         'contact_prediction',
         'remote_homology',
         'fluorescence',
         'stability']
TASK_METRICS = {'secondary_structure': 'accuracy',
                'remote_homology': 'accuracy',
                'contact_prediction': 'precision_at_l5',
                'fluorescence': 'spearmanr',
                'stability': 'spearmanr'}
TASK_SPLITS_TEST = {'secondary_structure':  ['casp12', 'cb513', 'ts115'],
                    'remote_homology':  ['test_fold_holdout',
                                        'test_family_holdout',
                                        'test_superfamily_holdout'],
                    'contact_prediction':  ['test'],
                    'fluorescence': ['test'],
                    'stability': ['test']}
TASK_SPLITS_VALID = {'secondary_structure':  ['valid'],
                     'remote_homology':  ['valid'],
                     'contact_prediction':  ['valid'],
                     'fluorescence': ['valid'],
                     'stability': ['valid']}

def starts_with_task(f):
    return np.any([f.startswith(t) for t in TASKS])

def get_arg_list(results_dir='/export/home/tape/results/',
                 split='valid'):
    if split == 'valid':
        task_splits = TASK_SPLITS_VALID
    else:
        task_splits = TASK_SPLITS_TEST

    pretrain_files = os.listdir(results_dir)
    pretrain_files = [f for f in pretrain_files if starts_with_task(f)]

    arg_list = []
    for pretrain_file in pretrain_files:
        file_split = pretrain_file.split('_')
        model_type = 'transformer'

        if '_joint_mlm_profile' in pretrain_file:
            task = pretrain_file.split('_joint_mlm_profile')[0]
        elif '_profile_prediction' in pretrain_file:
            task = pretrain_file.split('_profile_prediction')[0]
        elif '_masked_language_modeling' in pretrain_file:
            task = pretrain_file.split('_masked_language_modeling')[0]
        else:
            task = pretrain_file.split('_none')[0]
        from_pretrained = os.path.join(results_dir, pretrain_file)

        with open(os.path.join(from_pretrained, 'args.json')) as json_file:
            json_data = json.load(json_file)

        orig_file = json_data['from_pretrained']
        if orig_file is None:
            pretrain_task = None
        else:
            base_file = orig_file.split('/')[0]
            pretrain_task = base_file.split('_transformer_')[0]

        for split in task_splits[task]:
            d = {'model_type': model_type,
                 'task': task,
                 'from_pretrained': from_pretrained,
                 'batch_size': 32,
                 'metrics': TASK_METRICS[task],
                 'split': split,
                 'orig_file': orig_file,
                 'pretrain_task': pretrain_task}
            if task == 'contact_prediction':
                d['batch_size'] = 8
            arg_list.append(d)
    return arg_list

def get_args(model_type='transformer',
             task='secondary_structure',
             from_pretrained='results/secondary_structure_transformer_20-08-18-19-18-09_334672/',
             batch_size=8,
             metrics='accuracy',
             split='test',
             max_sequence_length=None,
             **kwargs):
    base_parser = create_base_parser()
    parser = create_eval_parser(base_parser)
    arg_string  = f'{model_type} {task} {from_pretrained} --num_workers 0 '
    arg_string += f'--batch_size {batch_size} --metrics {metrics} --split {split} '
    if max_sequence_length is not None:
        arg_string += f'--max_sequence_length {max_sequence_length}'
    args = parser.parse_args(shlex.split(arg_string))
    return args

def unravel_list_dict(l):
    n = len(l)
    unique_keys = set()
    for d in l:
        for k in d:
            unique_keys.add(k)
    d = {}
    for k in unique_keys:
        d[k] = [None] * n
    for index, item in enumerate(l):
        for name, value in item.items():
            d[name][index] = value
    return d

def write_list(args=None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    arg_list = get_arg_list(args.results_dir, args.split)
    out_file = args.out_file

    for arg_dict in arg_list:
        print(f'-----Running: {arg_dict}')
        args = get_args(**arg_dict)
        metrics = run_eval(args)
        for name, value in metrics.items():
            arg_dict[name] = value

    write_dict = unravel_list_dict(arg_list)
    results_df = pd.DataFrame(write_dict)
    results_df.to_csv(out_file)

if __name__ == '__main__':
    write_list()
