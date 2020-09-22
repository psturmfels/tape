import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
os.chdir('/export/home/tape/profile_depths/')

from netsurfp2 import accuracy
from plot_accuracies import *

import sys
sys.path[0] = '/export/home/tape'
from tape.utils import *

with open('cb513_netsurf.pkl', 'rb') as handle:
    predictions = pickle.load(handle)

profiles = np.load('netsurfp2/CB513_HHblits.npz')
ids = profiles['pdbids']
predictions['q3'].shape

accuracies = accuracy(predictions['labels'], predictions['q3'])
index_sorted = np.argsort(accuracies)

sorted_inputs = profiles['data'][index_sorted]
sorted_predictions = predictions['q3'][index_sorted]

plt.scatter(np.arange(len(index_sorted)), accuracies[index_sorted])

sorted_inputs.shape

mean_first = np.sum(sorted_inputs[:, :, 47], axis=1) / np.sum(sorted_inputs[:, :, 47] != 0.0, axis=1)
mean_second = np.sum(sorted_inputs[:, :, 48], axis=1) / np.sum(sorted_inputs[:, :, 48] != 0.0, axis=1)
mean_third = np.sum(sorted_inputs[:, :, 49], axis=1) / np.sum(sorted_inputs[:, :, 49] != 0.0, axis=1)
plt.scatter(mean_first, accuracies[index_sorted])
plt.scatter(mean_second, accuracies[index_sorted])
plt.scatter(mean_third, accuracies[index_sorted])

protein_lengths = np.sum(sorted_inputs[:, :, 50], axis=-1)

sparsity_pattern = np.sum(sorted_inputs[:, :, 20:40] != 0.0, axis=(1, 2)) / protein_lengths
plt.scatter(protein_lengths, accuracies[index_sorted])
plt.scatter(sparsity_pattern, accuracies[index_sorted])

reconstructed_netsurf_predictions = np.load('cb513_uniclust30_netsurf.npz')['q3']
reconstructed_netsurf_keys = list(sorted(list(set(ids))))
reconstructed_netsurf_dict = dict(zip(reconstructed_netsurf_keys, range(len(reconstructed_netsurf_keys))))
reconstructed_netsurf_predictions = np.stack([reconstructed_netsurf_predictions[reconstructed_netsurf_dict[id]] for id in ids], axis=0)
reconstructed_netsurf_accuracies = accuracy(predictions['labels'], reconstructed_netsurf_predictions)

plt.scatter(accuracies, reconstructed_netsurf_accuracies)

stats = []
for id in ids:
    file_name = f'cb513_uniclust30/{id}/{id}_PROFILE.hhm'
    stat = get_stats(file_name)
    stats.append(stat)
ids[0]
stats[0]

plt.scatter([stat['neff'] for stat in stats],
            reconstructed_netsurf_accuracies,
            c=[stat['lower'] for stat in stats])

plt.scatter([stat['num_seq'] for stat in stats],
            reconstructed_netsurf_accuracies,
            c=[stat['lower'] for stat in stats])
plt.scatter(np.linspace(2, 6, 100), [np.mean(reconstructed_netsurf_accuracies[[stat['neff'] < x for stat in stats]]) for x in np.linspace(2, 6, 100)])
plt.scatter(np.linspace(5, 1000, 1000), [np.mean(reconstructed_netsurf_accuracies[[stat['num_seq'] < x for stat in stats]]) for x in np.linspace(5, 1000, 1000)])


np.mean(reconstructed_netsurf_accuracies[[stat['num_seq'] < 40 for stat in stats]])


transformer_accuracies = [accuracy_single(batch) for batch in transformer_results]
transformer_ids = [a['id'].decode() for a in dataset.data]
transformer_dict = dict(zip(transformer_ids, transformer_accuracies))

with open('/export/home/tape/results/epoch34/secondary_structure_profile_prediction_transformer_full/cb513.pkl', 'rb') as handle:
    transformer_results = pickle.load(handle)
    transformer_results = transformer_results[1]

dataset = setup_dataset(task='secondary_structure',
                        data_dir='/export/home/tape/data',
                        split='cb513')

transformer_accuracies = [accuracy_single(batch) for batch in transformer_results]
transformer_ids = [a['id'].decode() for a in dataset.data]
transformer_dict = dict(zip(transformer_ids, transformer_accuracies))
transformer_accuracies = [transformer_dict[id] for id in ids]
transformer_accuracies = np.array(transformer_accuracies)

plt.scatter(np.linspace(2, 6, 100), [np.mean(transformer_accuracies[[stat['neff'] < x for stat in stats]]) for x in np.linspace(2, 6, 100)])
plt.scatter(np.linspace(5, 1000, 1000), [np.mean(transformer_accuracies[[stat['num_seq'] < x for stat in stats]]) for x in np.linspace(5, 1000, 1000)])
np.mean(transformer_accuracies[[stat['num_seq'] < 40 for stat in stats]])


plt.scatter([stat['neff'] for stat in stats],
            transformer_accuracies,
            c=[stat['lower'] for stat in stats])
