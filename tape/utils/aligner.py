import sys; sys.path[0] = '/export/home/tape'
import urllib
import pickle
import numpy as np
from tqdm import tqdm
from tape.tokenizers import BPETokenizer

class Aligner(object):
    def __init__(self,
                 blosum_url='https://www.ncbi.nlm.nih.gov/Class/BLAST/BLOSUM62.txt'):
        response = urllib.request.urlopen(blosum_url)
        data = []
        for line in response:
            line = line.decode('utf-8').strip()
            if line.startswith('#'):
                continue
            data.append(line)
        base_pairs = data[0].split()
        base_pair_index = dict([(base_pairs[index], index) for index in range(len(base_pairs))])

        values = [row.split()[1:] for row in data[1:]]
        values = [[int(x) for x in row] for row in values]
        values = np.array(values)
        self.alignment_indices = base_pair_index
        self.alignment_matrix = values

    def score(self,
              seq1,
              seq2,
              normalize_by_length=True):
        if len(seq1) != len(seq2):
            raise ValueError('This function assumes the sequences have the '
                             'same length. For variable length alignement, '
                             'you need to implement a DP aligner.')
        alignment_score = 0
        for base1, base2 in zip(seq1, seq2):
            index1 = self.alignment_indices['*']
            index2 = self.alignment_indices['*']
            if base1 in self.alignment_indices:
                index1 = self.alignment_indices[base1]
            if base2 in self.alignment_indices:
                index2 = self.alignment_indices[base2]

            alignment_score += self.alignment_matrix[index1, index2]
        if normalize_by_length:
            alignment_score /= len(seq1)
        return alignment_score

def main(argv):
    aligner = Aligner()
    with open('/export/home/tape/scripts/bpe/naive_10k.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    tokens = list(tokenizer.vocab.items())
    long_tokens = [token[0] for token in tokens if len(token[0]) == 5]
    similarity_matrix = np.zeros((len(long_tokens), len(long_tokens)))

    for i1, token1 in enumerate(tqdm(long_tokens)):
        for i2, token2 in enumerate(long_tokens):
            try:
                similarity_score = aligner.score(token1, token2)
            except ValueError:
                similarity_score = np.nan
            similarity_matrix[i1, i2] = similarity_score

    ordering = np.dstack(np.unravel_index(np.argsort(similarity_matrix.ravel()), similarity_matrix.shape))
    is_nan_mask = np.isnan(similarity_matrix[ordering[0, :, 0], ordering[0, :, 1]])
    indices_equal = ordering[0, :, 0] == ordering[0, :, 1]
    top_k = ordering[0, np.logical_and(~is_nan_mask, ~indices_equal)][-100:]

    for index in range(top_k.shape[0]):
        print('{:.4f}, {}, {}'.format(similarity_matrix[top_k[index, 0],
                                                        top_k[index, 1]],
                                      long_tokens[top_k[index, 0]],
                                      long_tokens[top_k[index, 1]]))

    filtered_indices = ordering[0, np.logical_and(~is_nan_mask, ~indices_equal)]
    flattened_matrix = similarity_matrix[filtered_indices[:, 0], filtered_indices[:, 1]]
    print(np.sum(flattened_matrix > 2), len(flattened_matrix))

if __name__ == '__main__':
    main()
