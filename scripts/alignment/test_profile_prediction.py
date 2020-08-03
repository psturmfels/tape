import os
import sys; sys.path[0] = '/export/home/tape'
import torch

from tape.utils import setup_dataset, setup_loader
from tape.registry import registry
from tape.models.modeling_utils import ProfileHead

def main():
    dataset = setup_dataset(task = 'profile_prediction',
                            data_dir = '/export/home/tape/data/alignment/pfam/test/',
                            split = 'train',
                            tokenizer = 'iupac')
    loader = setup_loader(dataset = dataset,
                          batch_size = 4,
                          local_rank = -1,
                          n_gpu = 1,
                          gradient_accumulation_steps = 1,
                          num_workers = 1)
    model = registry.get_task_model(model_name = 'transformer',
                                    task_name = 'profile_prediction')

    for i, batch in enumerate(loader):
        outputs = model(**batch)
        print(f'----Batch {i}----')
        print(batch['input_ids'].shape)
        print(batch['targets'].shape)
        print(outputs[0], outputs[1].shape)
        if i > 3:
            break

def test_head():
    inputs = torch.zeros(4, 32, 768, dtype=torch.float)
    head = ProfileHead(hidden_size=768)
    outputs = head(inputs)
    print(outputs)

def test_datasets():
    for split in ['train', 'holdout', 'valid']:
        dataset = setup_dataset(task = 'profile_prediction',
                                data_dir = '/export/home/tape/data/alignment/',
                                split = split,
                                tokenizer = 'iupac')
        print(dataset.data[len(dataset) - 1])
        print(f'Split {split} has {len(dataset)} elements')

if __name__ == '__main__':
    test_datasets()
