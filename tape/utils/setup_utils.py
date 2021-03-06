"""Utility functions to help setup the model, optimizer, distributed compute, etc.
"""
import typing
import logging
from pathlib import Path
import sys

## MARK: psturmfels custom code ##
##################################
import numpy as np
##################################
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from ..optimization import AdamW

from ..registry import registry

from .utils import get_effective_batch_size
from ._sampler import BucketBatchSampler

logger = logging.getLogger(__name__)


def setup_logging(local_rank: int,
                  save_path: typing.Optional[Path] = None,
                  log_level: typing.Union[str, int] = None) -> None:
    if log_level is None:
        level = logging.INFO
    elif isinstance(log_level, str):
        level = getattr(logging, log_level.upper())
    elif isinstance(log_level, int):
        level = log_level

    if local_rank not in (-1, 0):
        level = max(level, logging.WARN)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S")

    if not root_logger.hasHandlers():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if save_path is not None:
            file_handler = logging.FileHandler(save_path / 'log')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)


def setup_optimizer(model,
                    learning_rate: float):
    """Create the AdamW optimizer for the given model with the specified learning rate. Based on
    creation in the pytorch_transformers repository.

    Args:
        model (PreTrainedModel): The model for which to create an optimizer
        learning_rate (float): Default learning rate to use when creating the optimizer

    Returns:
        optimizer (AdamW): An AdamW optimizer

    """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def setup_dataset(task: str = 'masked_language_modeling',
                  data_dir: typing.Union[str, Path] = '/export/home/tape/data/alignment_indexed/',
                  split: str = 'train',
                  tokenizer: str = 'iupac',
                  dataset_fraction: float = None,
                  max_sequence_length: int = None) -> Dataset:
    task_spec = registry.get_task_spec(task)
    if max_sequence_length is not None:
        dataset = task_spec.dataset(data_dir, split, tokenizer, max_sequence_length=max_sequence_length)  # type: ignore
    else:
        dataset = task_spec.dataset(data_dir, split, tokenizer)
    if dataset_fraction is not None:
        dataset.data._num_examples = int(dataset_fraction * dataset.data._num_examples)
    return dataset


def setup_loader(dataset: Dataset,
                 batch_size: int = 32,
                 local_rank: int = -1,
                 n_gpu: int = 1,
                 gradient_accumulation_steps: int = 1,
                 num_workers: int = 1,
                 mask_fraction: typing.Optional[float] = None,
                 precomputed_key_file: str = '/export/home/tape/data/alignment_indexed/pfam/train_lengths.pkl') -> DataLoader:
    sampler = DistributedSampler(dataset) if local_rank != -1 else RandomSampler(dataset)
    batch_size = get_effective_batch_size(
        batch_size, local_rank, n_gpu, gradient_accumulation_steps) # * n_gpu
    # Note: Above the line multiplied by the number of GPUs. I suspect this is a mistake.
    # I've commented it out for now.

    # WARNING: this will fail if the primary sequence is not the first thing the dataset returns
    batch_sampler = BucketBatchSampler(
        sampler, batch_size, False, lambda x: len(x[0]), dataset,
        precomputed_key_file=precomputed_key_file)

    ## MARK: psturmfels custom code ##
    ##################################
    if mask_fraction is not None:
        def collate_fn(batch):
            batch_dict = dataset.collate_fn(batch)
            input_ids = batch_dict['input_ids']

            source_tensor = torch.ones_like(input_ids)
            selection_tensor = np.random.uniform(size=list(input_ids.shape))
            selection_tensor = selection_tensor < mask_fraction
            selection_tensor = torch.tensor(selection_tensor)
            input_ids.masked_scatter_(selection_tensor, source_tensor)
            batch_dict['input_ids'] = input_ids
            return batch_dict
    else:
        collate_fn = dataset.collate_fn
    ##################################
    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        ## MARK: psturmfels custom code ##
        ##################################
        collate_fn=collate_fn,  # type: ignore
        ##################################
        batch_sampler=batch_sampler)

    return loader


def setup_distributed(local_rank: int,
                      no_cuda: bool) -> typing.Tuple[torch.device, int, bool]:
    if local_rank != -1 and not no_cuda:
        torch.cuda.set_device(local_rank)
        device: torch.device = torch.device("cuda", local_rank)
        n_gpu = 1
        dist.init_process_group(backend="nccl")
    elif not torch.cuda.is_available() or no_cuda:
        device = torch.device("cpu")
        n_gpu = 1
    else:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()

    is_master = local_rank in (-1, 0)

    return device, n_gpu, is_master
