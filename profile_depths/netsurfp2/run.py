#!/usr/bin/env python
"""
NetSurfP-2.0 commandline frontend
"""

import os
import sys
import json
import time
import logging

from . import parse_fasta, preprocess, model, export_csv, export_npz

log = logging.getLogger('NetSurfP-2')
log.addHandler(logging.NullHandler())

def entry():
    import argparse

    # yapf: disable
    parser = argparse.ArgumentParser('netsurfp2', description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('method', help='Search method',
        choices=('mmseqs', 'hhblits'))
    parser.add_argument('model', help='Input model')
    parser.add_argument('inp', help='Input Fasta file')
    parser.add_argument('out', help='Output directory')
    # Output formats
    parser.add_argument('--npz', help='Export results as numpy format')
    parser.add_argument('--json', help='Export results as JSON format')
    parser.add_argument('--csv', help='Export results as CSV format')
    # Databases
    parser.add_argument('--hhdb', help='HHBlits Database',
        default='/data/Databases/hhsuite/uniclust30_2017_04/uniclust30_2017_04')
    parser.add_argument('--mmdb', help='MMseqs Database',
        default='/data/Databases/mmseqs/uniclust90_2017_04')
    parser.add_argument('--n_threads', default=16, type=int, help='number of jobs')
    # Other options
    parser.add_argument('--bs', help='Batch size', type=int, default=25)
    args = parser.parse_args()
    # yapf: enable

    logfmt = '%(asctime)s %(name)-12s: %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG,
        format=logfmt, datefmt='%Y-%m-%d %H:%M')

    with open(args.inp) as f:
        protlist = parse_fasta(f)

    log.info('Running {:,} sequences..'.format(len(protlist)))

    if args.method == 'hhblits':
        searcher = preprocess.HHblits(args.hhdb, n_threads=args.n_threads)
    elif args.method == 'mmseqs':
        searcher = preprocess.MMseqs(args.mmdb, n_threads=args.n_threads)

    computation_start = time.time()
    search_start = time.time()

    profiles = searcher(protlist, args.out)

    search_elapsed = time.time() - search_start
    log.info('Finished profiles ({:.1f} s, {:.1f} s per sequence)..'.format(search_elapsed, search_elapsed / len(protlist)))

    pred_start = time.time()
    nsp_model = model.TfGraphModel.load_graph(args.model)
    results = nsp_model.predict(profiles, args.out, batch_size=args.bs)
    pred_elapsed = time.time() - pred_start
    log.info('Finished predictions ({:.1f} s, {:.1f} s per sequence)..'.format(pred_elapsed, pred_elapsed / len(protlist)))

    if not any([args.npz, args.csv, args.json]):
        args.json = os.path.splitext(args.inp)[0] + '.nsp2.json'

    if args.json:
        with open(args.json, 'w') as fp:
            json.dump(results, fp, indent=2)

    if args.csv:
        with open(args.csv, 'w') as fp:
            export_csv(results, fp)

    if args.npz:
        export_npz(results, args.npz)

    time_elapsed = time.time() - computation_start

    log.info('Total time elapsed: {:.1f} s '.format(time_elapsed))
    log.info('Time per structure: {:.1f} s '.format(time_elapsed / len(protlist)))

    # model = load_model(modelpath)
    # preds = run(protlist, args.out, searcher, model)
    # print(preds)
    # preds = NetsurfpSession(searcher, modelpath)(protlist, args.out)

if __name__ == '__main__':
    entry()
