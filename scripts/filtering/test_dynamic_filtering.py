import os
import sys; sys.path[0] = '/export/home/tape/'
import torch
from tape.datasets import *
from tape.utils import setup_dataset, setup_loader
from tape.utils._sampler import *
from bisect import bisect_left
from tqdm import tqdm

dataset = setup_dataset(task='masked_language_modeling', split='valid', tokenizer='iupac', data_dir='/export/home/tape/data/')
loader = setup_loader(dataset = dataset,
                      batch_size = 400,
                      local_rank=-1,
                      n_gpu = 1,
                      gradient_accumulation_steps = 1,
                      num_workers = 2,
                      max_sequence_length = 300)

max = 0
for batch in tqdm(loader):
    if batch['input_ids'].shape[1] > max:
        max = batch['input_ids'].shape[1]
        print(max)
