"""
This file runs the vanilla BPE algorithm on the PFAM protein sequences
contained in this repository, for a specified number of iterations.

This file is essentially copied from:
https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py
"""
import os
os.chdir('/export/home/tape/')
import sys
import tape
import re
import copy
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter
from vocab_builder import VocabularyBuilder
from tape.tokenizers import BPETokenizer

from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_string('output_file',
                    '/export/home/tape/scripts/bpe/naive_10k.pkl',
                    'The directory to write the learned merge operations to.')
flags.DEFINE_integer('num_iterations',
                     10000,
                     'The number of iterations to run BPE for.')
flags.DEFINE_boolean('read_raw_vocab',
                     False,
                     'Set to true to not using a kmer-based vocabulary')
flags.DEFINE_string('vocab_file',
                    '/export/home/tape/scripts/bpe/vocab.pkl',
                    'The location of the saved kmer vocabulary')

def get_vocabulary(dataset):
    """
    Reads the entirety of the PFAM dataset (19G) into memory.
    """
    if FLAGS.read_raw_vocab:
        vocab = Counter()
        for i in range(len(dataset)):
            word = dataset.data[i]['primary']
            word = tuple(word)
            vocab[word] += 1
            if i % 100000 == 0:
                print('Reading vocabulary {}/{}'.format(i, len(dataset)))
    else:
        with open(FLAGS.vocab_file, 'rb') as vocab_file:
            vocab_builder = pickle.load(vocab_file)
            vocab = vocab_builder.vocab
    return vocab

def update_pair_statistics(pair, changed, stats, indices):
    """Minimally update the indices and frequency of symbol pairs
    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + second
    for j, word, old_word, freq in changed:

        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word) - 1 and old_word[i + 1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:
                    prev = old_word[i - 1:i + 1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word) - 2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second:
                        nex = old_word[i + 1:i + 3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i - 1:i + 1]
                stats[prev] += freq
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word) - 1 and word[i + 1] != new_pair:
                nex = word[i:i + 2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1

def get_pair_statistics(vocab):
    """Count frequency of all symbol pairs, and create index"""

    # data structure of pair frequencies
    stats = defaultdict(int)

    # index from pairs to words
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char

    return stats, indices

def replace_pair(pair, vocab, indices):
    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split())

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))

    return changes

def prune_stats(stats, big_stats, threshold):
    """Prune statistics dict for efficiency of max()
    The frequency of a symbol pair never increases, so pruning is generally safe
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    for item, freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0:
                big_stats[item] += freq
            else:
                big_stats[item] = freq

def learn_bpe(vocab, num_iterations, min_frequency=5, verbose=True):
    symbol_frequencies = []
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    print('Getting pair statistics...')
    stats, indices = get_pair_statistics(sorted_vocab)
    big_stats = copy.deepcopy(stats)

    print('Iterating...')
    threshold = max(stats.values()) / 10
    for i in range(num_iterations):
        if stats:
            most_frequent = max(stats, key=lambda x: (stats[x], x))

        # we probably missed the best pair because of pruning; go back to full statistics
        if not stats or (i and stats[most_frequent] < threshold):
            prune_stats(stats, big_stats, threshold)
            stats = copy.deepcopy(big_stats)
            most_frequent = max(stats, key=lambda x: (stats[x], x))
            # threshold is inspired by Zipfian assumption, but should only affect speed
            threshold = stats[most_frequent] * i/(i+10000.0)
            prune_stats(stats, big_stats, threshold)

        if stats[most_frequent] < min_frequency:
            sys.stderr.write('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break

        if verbose:
            print('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i,
                                                                         most_frequent[0],
                                                                         most_frequent[1],
                                                                         stats[most_frequent]))
        symbol_frequencies.append((most_frequent[0], most_frequent[1], stats[most_frequent]))
        print('Updating pair statistics at iteration {}/{}'.format(i, num_iterations))
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)
        stats[most_frequent] = 0
        if not i % 100:
            prune_stats(stats, big_stats, threshold)
    return symbol_frequencies

def main(argv=None):
    print('Loading the dataset...')
    dataset = tape.datasets.MaskedLanguageModelingDataset(data_path='data/',
                                                          split='train',
                                                          in_memory=True)
    print('Getting the vocabulary...')
    vocab = get_vocabulary(dataset)
    print('Running byte pair encoding...')
    symbol_frequencies = learn_bpe(vocab, num_iterations=FLAGS.num_iterations)
    tokenizer = BPETokenizer(merge_operations=symbol_frequencies)

    with open(FLAGS.output_file, 'wb') as save_file:
        pickle.dump(tokenizer, save_file)

if __name__ == '__main__':
    app.run(main)
