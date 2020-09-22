
"""
NetSurfP-2.0 main functions

"""

import os
import re
import json
import random

import numpy as np

def accuracy(labels, predictions):
    valid_mask = np.sum(labels, axis=-1).astype(int)
    sequence_lengths = np.sum(valid_mask, axis=-1)

    index_predictions = np.argmax(predictions, axis=-1)
    index_labels = np.argmax(labels, axis=-1)

    match = (index_predictions == index_labels).astype(int)
    index_sum = np.sum(valid_mask * match, axis=-1)

    accuracy = index_sum / sequence_lengths.astype(float)
    return accuracy

def parse_fasta(filehandle):
    i = 0
    prots = {}

    safe_id = None
    for line in filehandle:
        if line[0] == '>':
            original_id = line[1:].strip()
            unsafe_id = original_id.split()[1]
            unsafe_id = unsafe_id.split('|')[0]
            safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', unsafe_id)
            safe_id = '{}_{:0>4d}'.format(safe_id, i)

            # unsafe_id = original_id.split(None, 1)[0]
            # safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', unsafe_id)
            # safe_id = '{:0>4d}_{}'.format(i, safe_id)
            i += 1

            if safe_id in prots:
                print(safe_id)
            prots[safe_id] = [original_id, '']
        else:
            fragment = line.strip().upper()
            fragment = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', fragment)
            prots[safe_id][-1] += fragment
    return prots


def export_csv(results, filehandle):
    import csv
    header = [
        'id', 'seq', 'n', 'rsa', 'asa',
        'q3', 'p[q3_H]', 'p[q3_E]', 'p[q3_C]',
        'q8', 'p[q8_G]', 'p[q8_H]', 'p[q8_I]', 'p[q8_B]',
        'p[q8_E]', 'p[q8_S]', 'p[q8_T]', 'p[q8_C]',
        'phi', 'psi', 'disorder',
    ]
    writer = csv.DictWriter(filehandle, header)
    writer.writeheader()

    for res in results:
        pid = res['id']
        if res['desc']:
            pid = res['desc'].split(None, 1)[0]

        for i, aa in enumerate(res['seq']):
            pos = {
                'id': pid,
                'seq': aa,
                'n': i + 1,
                'rsa': res['rsa'][i],
                'asa': res['asa'][i],
                'phi': res['phi'][i],
                'psi': res['psi'][i],
                'disorder': res['disorder'][i],
                'q3': res['q3'][i],
                'q8': res['q8'][i],
            }

            for q3c, q3p in zip('HEC', res['q3_prob'][i]):
                pos['p[q3_{}]'.format(q3c)] = q3p

            for q8c, q8p in zip('GHIBESTC', res['q8_prob'][i]):
                pos['p[q8_{}]'.format(q8c)] = q8p

            writer.writerow(pos)


def export_nsp(results, filehandle):
    """Export NetSurfP-1.1 compatible output."""
    print("""# For publication of results, please cite:
# ...
# ... (Coming Soon)
# ...
#
# Column 1: Class assignment - B for buried or E for Exposed - Threshold: 25% exposure, but not based on RSA
# Column 2: Amino acid
# Column 3: Sequence name
# Column 4: Amino acid number
# Column 5: Relative Surface Accessibility - RSA
# Column 6: Absolute Surface Accessibility
# Column 7: Not used
# Column 8: Probability for Alpha-Helix
# Column 9: Probability for Beta-strand
# Column 10: Probability for Coil""", file=filehandle)

    for res in results:
        pid = res['desc'].split(None, 1)[0]
        for i, aa in enumerate(res['seq']):
            row = [
                'B' if res['rsa'][i] < 0.25 else 'E',
                aa,
                pid[:18],
                i + 1,
                res['rsa'][i],
                res['asa'][i],
                0,
                res['q3_prob'][i][0],
                res['q3_prob'][i][1],
                res['q3_prob'][i][2],
            ]
            print(('{} {}  {:<18} {:>5d} {:8.3f} {:7.3f} '
                   '{:7.3f} {:7.3f} {:7.3f} {:7.3f}').format(*row), file=filehandle)

def convert_npz(results):
    keys = ['rsa', 'asa', 'phi', 'psi', 'disorder', 'q3', 'q8']
    longest = max(len(r['seq']) for r in results)

    out = {}

    for key in keys:
        n = 2
        if key in ('rsa', 'asa'):
            n = 1
        elif key == 'q3':
            n = 3
        elif key == 'q8':
            n = 8

        out[key] = np.zeros([len(results), longest, n])

        for i, res in enumerate(results):
            l = len(res['seq'])
            if key in ('rsa', 'asa'):
                out[key][i, :l, 0] = res[key]
            elif key in ('phi', 'psi'):
                out[key][i, :l, 0] = np.sin(np.array(res[key]) * np.pi / 180)
                out[key][i, :l, 1] = np.cos(np.array(res[key]) * np.pi / 180)
            elif key in ('q3', 'q8'):
                out[key][i, :l, :] = res[key + '_prob']
            elif key == 'disorder':
                out[key][i, :l, 0] = 1 - np.array(res[key])
                out[key][i, :l, 1] = res[key]
    return out

def export_npz(results, filename):
    out = convert_npz(results)
    np.savez_compressed(filename, **out)


class NetsurfpException(Exception):
    pass


class NetsurfpValueError(NetsurfpException, ValueError):
    pass
