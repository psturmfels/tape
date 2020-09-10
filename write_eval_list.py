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
             {'model_type': 'transformer', 'task': 'contact_prediction', 'split': 'test', 'pretrain_task': 'tape_baseline', 'precision_at_l5': 0.66},
             {'model_type': 'transformer', 'task': 'fluorescence', 'split': 'test', 'pretrain_task': 'tape_none', 'spearmanr': 0.22},
             {'model_type': 'transformer', 'task': 'fluorescence', 'split': 'test', 'pretrain_task': 'tape_mlm', 'spearmanr': 0.68},
             {'model_type': 'transformer', 'task': 'fluorescence', 'split': 'test', 'pretrain_task': 'tape_baseline', 'spearmanr': 0.67},
             {'model_type': 'transformer', 'task': 'stability', 'split': 'test', 'pretrain_task': 'tape_none', 'spearmanr': -0.06},
             {'model_type': 'transformer', 'task': 'stability', 'split': 'test', 'pretrain_task': 'tape_mlm', 'spearmanr': 0.73},
             {'model_type': 'transformer', 'task': 'stability', 'split': 'test', 'pretrain_task': 'tape_baseline', 'spearmanr': 0.73}]

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
        if '_joint_mlm_profile_transformer' in pretrain_file:
            task = pretrain_file.split('_joint_mlm_profile_transformer')[0]
        elif '_profile_prediction_transformer' in pretrain_file:
            task = pretrain_file.split('_profile_prediction_transformer')[0]
        elif '_masked_language_modeling_transformer' in pretrain_file:
            task = pretrain_file.split('_masked_language_modeling_transformer')[0]
        else:
            task = pretrain_file.split(f'_{model_type}_')[0]
        from_pretrained = os.path.join('results/', pretrain_file)

        with open(os.path.join(from_pretrained, 'args.json')) as json_file:
            json_data = json.load(json_file)

        orig_file = json_data['from_pretrained']
        if orig_file is None:
            pretrain_task = None
        else:
            pretrain_task = orig_file.split('_' + model_type + '_')[0].split('/')[-1]
            orig_file = orig_file.split('/home/tape/')[-1]

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
                d['batch_size'] = 4
            arg_list.append(d)
    return arg_list

def get_args(model_type='transformer',
             task='secondary_structure',
             from_pretrained='results/secondary_structure_transformer_20-08-18-19-18-09_334672/',
             batch_size=4,
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

def strip_df(df):
    orig_task_list = df.loc[[(isinstance(item, str) and '-15-' in item) for item in df['orig_file']], 'pretrain_task']
    orig_task_list = [task + '_long' for task in orig_task_list]
    df.loc[[(isinstance(item, str) and '-15-' in item) for item in df['orig_file']], 'pretrain_task'] = orig_task_list
    is_from_tape = [isinstance(task, str) and task.startswith('tape') for task in df['pretrain_task']]
    df['is_from_tape'] = is_from_tape
    df.loc[pd.isnull(df['pretrain_task']), 'pretrain_task'] = 'none'
    return df

def barplot_task(df,
                 task='secondary_structure',
                 custom_order=['none',
                               'tape_none',
                               'profile_prediction',
                               'joint_mlm_profile',
                               'tape_mlm',
                               'tape_baseline'],
                 custom_spacing=[0,
                                 1,
                                 2.4,
                                 3.4,
                                 4.4,
                                 5.8],
                ylim=(0.6, 0.8),
                label_size=14,
                title_size=16,
                suptitle_size=20,
                label_rotation=40,
                dpi=150,
                num_ticks=6,
                metric='accuracy'):
    splits = np.unique(df.loc[df['task'] == task, 'split'])
    fig, axs = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5), dpi=dpi)
    if len(splits) == 1:
        axs = [axs]
    colors = ['firebrick', 'darkblue', 'forestgreen', 'purple', 'orange']
    for split, ax, color in zip(splits, axs, colors):
        select_df = df[np.logical_and(df['task'] == task,
                                      df['split'] == split)]

        select_df = select_df.set_index('pretrain_task').reindex(custom_order).reset_index()
        ax.bar(custom_spacing, select_df[metric], color=color)
        ax.set_xticks(custom_spacing)
        ax.set_xticklabels(select_df['pretrain_task'], rotation=label_rotation, ha='right')
        ax.tick_params(labelsize=label_size)
        ax.set_yticks(np.linspace(ylim[0], ylim[1], num_ticks))
        ax.set_ylim(*ylim)
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['top'].set_linewidth(0.1)
        ax.spines['right'].set_linewidth(0.1)
        ax.set_title(f'split: {split}', fontsize=title_size)
    fig.suptitle(f'Task: {task}', fontsize=suptitle_size, y=1.1)
    fig.tight_layout()
    return fig, axs

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

    arg_list += TAPE_ROWS
    write_dict = unravel_list_dict(arg_list)
    results_df = pd.DataFrame(write_dict)
    results_df = strip_df(results_df)
    results_df.to_csv(out_file)

    # fig, ax = barplot_task(results_df, task='secondary_structure')
    # fig.savefig('figures/secondary_structure.png', dpi=150)
    #
    # fig, ax = barplot_task(results_df, task='remote_homology', ylim=(0.0, 1.0), num_ticks=9)
    # fig.savefig('figures/remote_homology.png', dpi=150)
    #
    # fig, ax = barplot_task(results_df, task='contact_prediction', ylim=(0.0, 0.7), metric='precision_at_l5')
    # fig.savefig('figures/contact_prediction.png', dpi=150)
    #
    # fig, ax = barplot_task(results_df, task='fluorescence', ylim=(0.0, 1.0), num_ticks=9)
    # fig.savefig('figures/fluorescence.png', dpi=150)
    #
    # fig, ax = barplot_task(results_df, task='stability', ylim=(0.0, 1.0), num_ticks=9)
    # fig.savefig('figures/stability.png', dpi=150)

if __name__ == '__main__':
    write_list()
