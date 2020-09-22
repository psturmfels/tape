
'''
NetSurfP-2.0 Post-processing of predictions

'''

import numpy as np


ASA_MAX = {
    'A': 110.2, 'C': 140.4, 'D': 144.1, 'E': 174.7, 'F': 200.7,
    'G': 78.7,  'H': 181.9, 'I': 185.0, 'K': 205.7, 'L': 183.1,
    'M': 200.1, 'N': 146.4, 'P': 141.9, 'Q': 178.6, 'R': 229.0,
    'S': 117.2, 'T': 138.7, 'V': 153.7, 'W': 240.5, 'Y': 213.7,
}


def get_processor(name):
    return globals()['process_' + name]


def process_interface(y, seq=None):
    return {'interface': y[:, 0].tolist()}


def process_disorder(y, seq=None):
    return {'disorder': y[:, 1].tolist()}


def angle(y):
    return (np.arctan2(y[:, 0], y[:, 1]) * (180 / np.pi)).tolist()


def process_phi(y, seq=None):
    return {'phi': angle(y)}


def process_psi(y, seq=None):
    return {'psi': angle(y)}


def process_rsa(y, seq):
    return {
        'rsa': y[:, 0].tolist(),
        'asa': [r * ASA_MAX.get(s, 0.) for r, s in zip(y[:, 0], seq)],
    }


process_isorsa = process_rsa


def process_q3(y, seq=None):
    return {
        'q3': ''.join('HEC'[s] for s in np.argmax(y, axis=-1)),
        'q3_prob': y.tolist(),
    }


def process_q8(y, seq=None):
    #preds['q8'] = ''.join('GHIBESTC'[s] for s in np.argmax(pred, axis=-1))
    return {
        'q8': ''.join('GHIBESTC'[s] for s in np.argmax(y, axis=-1)),
        'q8_prob': y.tolist(),
    }
