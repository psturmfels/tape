import os
os.chdir('/export/home/tape/')

import torch
import tape
from tape import utils

dataset = tape.datasets.MaskedLanguageModelingDataset(data_path='data/', split='train')
dataset.data[0]
len(dataset)


for i, item in enumerate(dataset.data):
    print(i, item)
    if i == 2:
        break

data_loader = utils.setup_loader(dataset=dataset,
                                 batch_size=4,
                                 local_rank=-1,
                                 n_gpu=1,
                                 gradient_accumulation_steps=1,
                                 num_workers=8)

for i, item in enumerate(data_loader):
    print(i, item)
    if i == 3:
        break

item['targets']
