"""
NetSurfP-2.0 model utilities

"""

import os
import re
import logging

import numpy as np
import tensorflow as tf

from . import postprocess
from .preprocess import parse_hhm
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
log = logging.getLogger('NetSurfP-2')
log.addHandler(logging.NullHandler())


class TfGraphModel:

    def __init__(self, graph):
        self.graph = graph

    @classmethod
    def load_graph(cls, filename):
        with tf.io.gfile.GFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='NSP2')

        return cls(graph)

    def predict(self, profiles, output_dir='', batch_size=50, n_threads=0):
        protlens = {pid: len(profiles[pid]['seq']) for pid in profiles}
        protids = sorted(profiles, key=protlens.get)
        protlens = [protlens[pid] for pid in protids]

        # print (protlens)

        n_features = 51#profiles[protids[0]]['profile'].shape[1]

        results = {}

        for i, protid in enumerate(protids):
            if (i + 1) % 1000 == 0:
                log.info('   ..predicted {:,} sequences'.format(i + 1))

            #j is the in-batch iterater (from 0 to batch_size)
            j = i % batch_size
            if j == 0:
                _longest = max(protlens[i:i+batch_size])
                _bs = min([len(profiles) - i, batch_size])
                _profile_batch = np.zeros((_bs, _longest, n_features))

            if profiles[protid]['profile'] is not None:
                profile = profiles[protid]['profile']
            else:
                prot_dir = os.path.abspath(os.path.join(output_dir, protid))
                hhm_name = os.path.join(prot_dir, protid + '_PROFILE.hhm')
                with open(hhm_name) as fp:
                    profile = parse_hhm(fp, seq=profiles[protid]['seq'])['profile']

            _profile_batch[j, :profile.shape[0], :] = profile

            if j == _bs - 1:
                i_bs = _bs
                raw_predictions = None
                while raw_predictions is None:
                    try:
                        raw_predictions = self._predict_array(_profile_batch,
                                                              batch_size=i_bs,
                                                              n_threads=n_threads)
                    except tf.errors.ResourceExhaustedError:
                        i_bs = i_bs // 2
                        if i_bs < 1:
                            raise
                        log.warning('ResourceExhaustedError thrown, running batch with batch_size={}'.format(i_bs))

                for k in range(_bs):
                    h = i - _bs + k + 1
                    l = protlens[h]
                    protid = protids[h]
                    preds = {}
                    for output_name in raw_predictions:
                        processor = postprocess.get_processor(output_name)

                        raw_pred = raw_predictions[output_name][k, :l]
                        preds.update(processor(raw_pred, profiles[protid]['seq']))

                    preds['id']     = protid
                    preds['seq']    = profiles[protid]['seq']
                    preds['desc']   = profiles[protid]['desc']
                    preds['method'] = profiles[protid]['method']
                    results[protid] = preds

        results = [results[pid] for pid in sorted(results)]


        return results

    def _predict_array(self, profile_array, batch_size=50, n_threads=0):
        feed_dict = {}
        outputs = []
        y = []
        for op in self.graph.get_operations():
            if op.name.endswith('keras_learning_phase'):
                feed_dict[self.graph.get_tensor_by_name(op.name + ':0')] = 0
                continue

            mx = re.match(r'NSP2/y_(\w+)/Reshape_1$', op.name)
            if mx:
                outputs.append(mx.group(1))
                y.append(self.graph.get_tensor_by_name(op.name + ':0'))

        x = self.graph.get_tensor_by_name('NSP2/x:0')
        x_mask = self.graph.get_tensor_by_name('NSP2/x_mask:0')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = n_threads
        config.inter_op_parallelism_threads = n_threads

        with tf.Session(graph=self.graph, config=config) as sess:
            y_out = {o: [] for o in outputs}
            for i in tqdm(range(0, profile_array.shape[0], batch_size)):
                feed_dict[x] = profile_array[i:i+batch_size]
                feed_dict[x_mask] = profile_array[i:i+batch_size, :, -1:]
                _y_out = sess.run(y, feed_dict=feed_dict)

                for o, p in zip(outputs, _y_out):
                    y_out[o].append(p)

        for o, p in y_out.items():
            y_out[o] = np.concatenate(p)

        return y_out
