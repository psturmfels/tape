#!/opt/conda/bin/python
# -*- coding: utf-8 -*-
import re
import sys
import argparse
import shlex
import json
import numpy as np
import pandas as pd
from tape.main import *

TASKS = ['secondary_structure',
         'contact_prediction',
         'remote_homology']
TASK_SPLITS = {'secondary_structure':  ['casp12', 'cb513', 'ts115'],
               'remote_homology':  ['test_fold_holdout',
                                   'test_family_holdout',
                                   'test_superfamily_holdout'],
               'contact_prediction':  ['test']}
TAPE_ROWS = [{'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'cb513', 'pretrain_task': 'tape_none', 'accuracy': 0.70},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'casp12', 'pretrain_task': 'tape_none', 'accuracy': 0.68},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'ts115', 'pretrain_task': 'tape_none', 'accuracy': 0.73},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'cb513', 'pretrain_task': 'tape_mlm', 'accuracy': 0.73},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'casp12', 'pretrain_task': 'tape_mlm', 'accuracy': 0.71},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'ts115', 'pretrain_task': 'tape_mlm', 'accuracy': 0.77},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'cb513', 'pretrain_task': 'tape_baseline', 'accuracy': 0.80},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'casp12', 'pretrain_task': 'tape_baseline', 'accuracy': 0.76},
             {'model_type': 'transformer', 'task': 'secondary_structure', 'split': 'ts115', 'pretrain_task': 'tape_baseline', 'accuracy': 0.81},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_fold_holdout', 'pretrain_task': 'tape_none', 'accuracy': 0.09},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_family_holdout', 'pretrain_task': 'tape_none', 'accuracy': 0.31},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_superfamily_holdout', 'pretrain_task': 'tape_none', 'accuracy': 0.07},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_fold_holdout', 'pretrain_task': 'tape_mlm', 'accuracy': 0.21},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_family_holdout', 'pretrain_task': 'tape_mlm', 'accuracy': 0.88},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_superfamily_holdout', 'pretrain_task': 'tape_mlm', 'accuracy': 0.34},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_fold_holdout', 'pretrain_task': 'tape_baseline', 'accuracy': 0.26},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_family_holdout', 'pretrain_task': 'tape_baseline', 'accuracy': 0.92},
             {'model_type': 'transformer', 'task': 'remote_homology', 'split': 'test_superfamily_holdout', 'pretrain_task': 'tape_baseline', 'accuracy': 0.43},
             {'model_type': 'transformer', 'task': 'contact_prediction', 'split': 'test', 'pretrain_task': 'tape_none', 'precision_at_l5': 0.40},
             {'model_type': 'transformer', 'task': 'contact_prediction', 'split': 'test', 'pretrain_task': 'tape_mlm', 'precision_at_l5': 0.46},
             {'model_type': 'transformer', 'task': 'contact_prediction', 'split': 'test', 'pretrain_task': 'tape_baseline', 'precision_at_l5': 0.66}]

def starts_with_task(f):
    return np.any([f.startswith(t) for t in TASKS])

def get_arg_list(results_dir='/export/home/tape/results/'):
    pretrain_files = os.listdir(results_dir)
    pretrain_files = [f for f in pretrain_files if starts_with_task(f)]

    arg_list = []
    for pretrain_file in pretrain_files:
        file_split = pretrain_file.split('_')
        task = file_split[0] + '_' + file_split[1]
        model_type = file_split[2]
        from_pretrained = os.path.join('results/', pretrain_file)

        with open(os.path.join(from_pretrained, 'args.json')) as json_file:
            json_data = json.load(json_file)

        orig_file = json_data['from_pretrained']
        if orig_file is None:
            pretrain_task = None
        else:
            pretrain_task = orig_file.split('_' + model_type + '_')[0].split('/')[-1]
            orig_file = orig_file.split('/home/tape/')[-1]

        for split in TASK_SPLITS[task]:
            arg_list.append({'model_type': model_type,
                             'task': task,
                             'from_pretrained': from_pretrained,
                             'batch_size': 32,
                             'metrics': 'accuracy',
                             'split': split,
                             'max_sequence_length': 270,
                             'orig_file': orig_file,
                             'pretrain_task': pretrain_task})
    return arg_list

def get_args(model_type='transformer',
             task='secondary_structure',
             from_pretrained='results/secondary_structure_transformer_20-08-18-19-18-09_334672/',
             batch_size=32,
             metrics='accuracy',
             split='test',
             max_sequence_length=270,
             **kwargs):
    base_parser = create_base_parser()
    parser = create_eval_parser(base_parser)
    arg_string  = f'{model_type} {task} {from_pretrained} '
    arg_string += f'--batch_size {batch_size} --metrics {metrics} --split {split} '
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

def write_list():
    arg_list = get_arg_list()

    for arg_dict in arg_list:
        print(f'-----Running: {arg_dict}')
        args = get_args(**arg_dict)
        metrics = run_eval(args)
        for name, value in metrics.items():
            arg_dict[name] = value

    arg_list += TAPE_ROWS
    write_dict = unravel_list_dict(arg_list)
    results_df = pd.DataFrame(write_dict)
    results_df.to_csv('/export/home/tape/eval_results.csv')

if __name__ == '__main__':
    write_list()
