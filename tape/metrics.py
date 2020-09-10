from typing import Sequence, Union, Any, Tuple, Dict
import numpy as np
import scipy.stats
import torch
from scipy.special import softmax

from .registry import registry

def strip_save_outputs(save_outputs: Sequence[Dict[str, Any]]) -> Tuple[Sequence[float], Sequence[float]]:
    target = [el['target'] for el in save_outputs]
    prediction = [el['prediction'] for el in save_outputs]
    return target, prediction

@registry.register_metric('mse')
def mean_squared_error(save_outputs: Sequence[Dict[str, Any]]) -> float:
    target, prediction = strip_save_outputs(save_outputs)
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))


@registry.register_metric('mae')
def mean_absolute_error(save_outputs: Sequence[Dict[str, Any]]) -> float:
    target, prediction = strip_save_outputs(save_outputs)
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))


@registry.register_metric('spearmanr')
def spearmanr(save_outputs: Sequence[Dict[str, Any]]) -> float:
    target, prediction = strip_save_outputs(save_outputs)
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.spearmanr(target_array, prediction_array).correlation


@registry.register_metric('accuracy')
def accuracy(save_outputs: Sequence[Dict[str, Any]]) -> float:
    target, prediction = strip_save_outputs(save_outputs)
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total

@registry.register_metric('precision_at_l5')
def precision_at_l5(save_outputs: Sequence[Dict[str, Any]]) -> float:
    target, prediction = strip_save_outputs(save_outputs)
    sequence_lengths = [el['length'] for el in save_outputs]

    ignore_index = -1

    correct = 0.0
    total = 0.0
    for length, pred, label in zip(sequence_lengths, prediction, target):
        prob = softmax(pred, axis=2)[:, :, 1]

        mask = label != ignore_index
        seqpos = np.arange(mask.shape[0])
        y_ind, x_ind = np.meshgrid(seqpos, seqpos)
        mask &= ((y_ind - x_ind) >=6)
        mask = mask.astype(prob.dtype)

        masked_prob = np.reshape(prob * mask, -1)
        most_likely_indices = np.argsort(masked_prob)[-(length // 5):]
        selected = np.reshape(label, -1)[most_likely_indices]
        correct += np.sum(selected).astype(float)
        total += float(selected.size)
    return correct / total

# def precision_single(pred,
#                      label,
#                      length,
#                      ignore_index=-1):
#     prob = softmax(pred, axis=2)[:, :, 1]
#
#     mask = label != ignore_index
#     seqpos = np.arange(mask.shape[0])
#     y_ind, x_ind = np.meshgrid(seqpos, seqpos)
#     mask &= ((y_ind - x_ind) >=6)
#     mask = mask.astype(prob.dtype)
#
#     masked_prob = np.reshape(prob * mask, -1)
#     most_likely_indices = np.argsort(masked_prob)[-(length // 5):]
#     selected = np.reshape(label, -1)[most_likely_indices]
#     return np.sum(selected) / float(selected.size)
#
# def accuracy_single(pred,
#                     label,
#                     ignore_index=-1):
#     pred_array  = score.argmax(-1)
#     mask = label != ignore_index=-1
#     is_correct = label[mask] == pred_array[mask]
#     return is_correct.sum() / float(is_correct.size)
