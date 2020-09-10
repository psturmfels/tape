import os
import sys
import argparse
from tape.registry import registry
from tape.main import run_train_distributed, create_train_parser, create_base_parser, create_distributed_parser

def create_parser():
    base_parser = create_base_parser()
    distributed_parser = create_distributed_parser(base_parser)
    distributed_train_parser = create_train_parser(distributed_parser)
    model_type_action = distributed_train_parser._actions[20]
    model_type_action.choices.append('all')
    return distributed_train_parser

def run_all(args = None):
    if args is None:
        parser = create_parser()
        args = parser.parse_args()

    if args.from_pretrained == 'joint_all':
        files = [os.path.join('results/', file) for file in os.listdir('results/') \
                 if file.startswith('joint_mlm_profile')]
    else:
        files = [args.from_pretrained]

    if args.task == 'all':
        tasks = ['contact_prediction',]
                 # 'fluorescence',
                 # 'stability',
                 # 'remote_homology',
                 # 'secondary_structure']
    else:
        tasks = [args.task]

    for task in tasks:
        for pretrained in files:
            args.task = task
            args.from_pretrained = pretrained
            args.exp_name = f'{task}_' + pretrained.split('results/')[-1]
            if task == 'contact_prediction':
                args.gradient_accumulation_steps = 16
                args.batch_size = 128
                args.max_sequence_length = 800
            else:
                args.gradient_accumulation_steps = 3
                args.batch_size = 512
                args.max_sequence_length = None
            run_train_distributed(args)

if __name__ == '__main__':
    run_all()
