import os
import time
os.chdir('/export/home/tape/')

import torch
import pickle
import tape
from tape import utils

from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('kmer_length', 10, 'Default size of shredded kmers')
flags.DEFINE_integer('kmer_overlap', 5, 'Amount that kmers overlap by')

class VocabularyBuilder(object):
    def __init__(self, vocab=None):
        self.vocab = vocab

    def _shred_to_kmer(self,
                       sequence,
                       k=10,
                       overlap=5):
        kmers = []
        increment = k - overlap
        for i in range(0, len(sequence), increment):
            if i + k >= len(sequence):
                kmer = sequence[-k:]
                kmers.append(kmer)
                break
            else:
                kmer = sequence[i:i + k]
                kmers.append(kmer)
        return kmers

    def build_vocabulary(self,
                         dataset,
                         key='primary',
                         k=10,
                         overlap=5):
        print("Building vocabulary. This might take a while...")
        vocab = {}
        for i in range(len(dataset)):
            if i % 100000 == 0:
                print('Iteration {}/{}'.format(i, len(dataset)))
            sequence = dataset.data[i][key]
            kmers = self._shred_to_kmer(sequence,
                                        k=k,
                                        overlap=overlap)
            for kmer in kmers:
                if kmer in vocab:
                    vocab[kmer] += 1
                else:
                    vocab[kmer] = 1
        self.vocab = vocab

def main(argv=None):
    dataset = tape.datasets.MaskedLanguageModelingDataset(data_path='data/',
                                                          split='train')
    builder = VocabularyBuilder()
    builder.build_vocabulary(dataset,
                             k=FLAGS.kmer_length,
                             overlap=FLAGS.kmer_overlap)
    with open('/export/home/tape/scripts/bpe/vocab.pkl', 'wb') as save_file:
        pickle.dump(builder, save_file)

if __name__ == '__main__':
    app.run(main)
