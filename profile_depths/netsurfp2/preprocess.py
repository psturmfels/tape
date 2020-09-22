#!/usr/bin/env python
"""
Wrappers for database searching
"""

import os
import pickle
import sqlite3
import subprocess
import collections

import numpy as np

import logging
log = logging.getLogger('NetSurfP-2')
log.addHandler(logging.NullHandler())


class HHblits:

    BIN = 'hhblits'
    name = 'HHblits'

    def __init__(self, db, hh_iters=2, n_threads=None, use_existing=False):
        self.db = db
        self.hh_iters = str(int(hh_iters))
        self.n_threads = int(n_threads) if n_threads else None
        self.use_existing = use_existing

    def __call__(self, protlist, output_dir):
        profiles = {}
        for protid, (desc, seq) in protlist.items():
            hhm, a3m = self._search(protid, seq, output_dir)
            profiles[protid] = parse_hhm(iter(hhm.splitlines()))
            profiles[protid]['desc'] = desc
            profiles[protid]['method'] = self.name

        return profiles

    def _search(self, protid, seq, output_dir):
        output_dir = os.path.abspath(os.path.join(output_dir, protid))
        os.makedirs(output_dir, exist_ok=True)

        faa_name = os.path.join(output_dir, protid + '_INPUT.fasta')
        hhm_name = os.path.join(output_dir, protid + '_PROFILE.hhm')
        a3m_name = os.path.join(output_dir, protid + '_MSA.a3m')
        log_name = os.path.join(output_dir, protid + '_hhblits.log')

        if not os.path.exists(hhm_name) or not self.use_existing:
            with open(faa_name, 'w') as faa_handle:
                print('>' + protid, file=faa_handle)
                print(seq, file=faa_handle)

            cmd = [
                self.BIN, '-i', faa_name, '-o', '/dev/null', '-ohhm', hhm_name,
                '-n', self.hh_iters, '-d', self.db, '-oa3m', a3m_name,
            ] # yapf: disable
            if self.n_threads:
                cmd.extend(['-cpu', str(self.n_threads)])

            with open(log_name, 'w') as logfp:
                skw = dict(stdout=logfp, stderr=logfp)
                p = subprocess.run(cmd, universal_newlines=True, **skw)

            p.check_returncode()

        with open(hhm_name) as f:
            hhm = f.read()
        with open(a3m_name) as f:
            a3m = f.read()

        return hhm, a3m


class MMseqs:

    BIN = 'mmseqs'
    HHMAKE_BIN = 'hhmake'
    name = 'MMseqs2'

    def __init__(self, db, n_threads=None, split_db=0):
        self.db = db
        self.n_threads = int(n_threads) if n_threads else None
        self.split_db = int(split_db) if split_db else 0

    def __call__(self, protlist, output_dir, parse_to_mem=None):
        input_protlist = {}
        profiles = {}

        log.info('Writing fasta files..')
        for protid, (desc, seq) in protlist.items():
            #Output directory for input protein
            prot_dir = os.path.abspath(os.path.join(output_dir, protid))
            os.makedirs(prot_dir, exist_ok=True)

            #Write fasta input (Not used by MMseqs, but here for compatability)
            faa_name = os.path.join(prot_dir, protid + '_INPUT.fasta')
            with open(faa_name, 'w') as faa_handle:
                print('>' + protid, file=faa_handle)
                print(seq, file=faa_handle)

            input_protlist[protid] = seq

        mmdir = os.path.join(output_dir, 'mmseqs_files')
        os.makedirs(mmdir, exist_ok=True)
        mm_faa = os.path.join(mmdir, 'in.fasta')
        mm_db = os.path.join(mmdir, 'in.mmdb')

        mmlogfp = open(os.path.join(mmdir, 'log.txt'), 'w')
        skw = dict(stdout=mmlogfp, stderr=mmlogfp, check=True)

        if not os.path.isfile(mm_db):
            with open(mm_faa, 'w') as f:
                for protid, seq in sorted(input_protlist.items()):
                    print('>' + protid, file=f)
                    print(seq, file=f)

            subprocess.run([MMseqs.BIN, 'createdb', mm_faa, mm_db], **skw)

        #mmtmp = os.path.join(mmdir, 'tmp')
        mmmsa = os.path.join(mmdir, 'out.mm_msa')
        mmres = os.path.join(mmdir, 'out.mm_search')

        import tempfile

        if not os.path.isfile(mmmsa):
            #os.makedirs(mmtmp, exist_ok=True)

            with tempfile.TemporaryDirectory() as mmtmp:
                search_cmd = [
                    MMseqs.BIN, 'search', mm_db, self.db, mmres, mmtmp,
                    '--num-iterations', '2', '--max-seqs', '2000'
                ]
                if self.n_threads:
                    search_cmd.extend(['--threads', str(self.n_threads)])
                if self.split_db:
                    search_cmd.extend(['--split', str(self.split_db)])

                log.info('Running `{}`'.format(' '.join(search_cmd)))
                subprocess.run(search_cmd, **skw) #yapf: disable

            msa_cmd = [MMseqs.BIN, 'result2msa', mm_db, self.db, mmres, mmmsa]
            log.info('Running `{}`'.format(' '.join(msa_cmd)))
            subprocess.run(msa_cmd, **skw) #yapf: disable

        if parse_to_mem is None:
            parse_to_mem = True
            if len(protlist) > 10000:
                parse_to_mem = False

        log.info('Parsing MMseqs2 MSA database')
        with open(mmmsa) as fdat, open(mmmsa + '.index') as fidx:
            import multiprocessing

            pool = multiprocessing.Pool(self.n_threads)
            promises = collections.OrderedDict()

            for i, line in enumerate(fidx, 1):
                if i % 5000 == 0:
                    log.info('    ..parsed {:,} MSAs'.format(i))

                msa_id, start, length = line.strip().split('\t')
                fdat.seek(int(start))
                msa = fdat.read(int(length) - 1)
                protid = msa.split(None, 1)[0][1:]

                #Happens if protein is longer that 32K.
                if protid not in protlist:
                    continue

                seq = protlist[protid][1]

                promises[protid] = pool.apply_async(
                    process_msa, (msa, protid, output_dir, seq, parse_to_mem))

        log.info('Creating HHM profiles')
        for i, (protid, promise) in enumerate(sorted(promises.items()), 1):
            if i % 5000 == 0:
                log.info('    ..created {:,} profiles'.format(i))

            profiles[protid] = promise.get()
            profiles[protid]['desc'] = protlist[protid][0]
            profiles[protid]['method'] = self.name

        return profiles


def process_msa(msa, protid, output_dir, seq, parse_to_mem):
    prot_dir = os.path.abspath(os.path.join(output_dir, protid))
    hhm_name = os.path.join(prot_dir, protid + '_PROFILE.hhm')
    msa_name = os.path.join(prot_dir, protid + '_MSA.fasta')

    if os.path.isfile(hhm_name) and os.path.isfile(msa_name):
        with open(hhm_name) as fp:
            res = parse_hhm(fp, seq=seq)
        if not parse_to_mem:
            res['profile'] = None
        return res

    with open(msa_name, 'w') as f:
        f.write(msa)

    proc = subprocess.run([
            MMseqs.HHMAKE_BIN, '-i', msa_name, '-v', '0', '-o',
            'stdout', '-M', 'first'],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE) # yapf: disable

    hhm = proc.stdout
    with open(hhm_name, 'w') as f:
        f.write(hhm)

    res = parse_hhm(iter(hhm.splitlines()), seq=seq)
    if not parse_to_mem:
        res['profile'] = None

    return res


def freq(freqstr):
    if freqstr == '*':
        return 0.
    p = 2**(int(freqstr) / -1000)
    assert 0 <= p <= 1.0
    return p


def parse_hhm(hhmfp, seq=None):
    neff = None
    for line in hhmfp:
        if line[0:4] == 'NEFF' and neff is None:
            neff = float(line[4:].strip())
        if line[0:8] == 'HMM    A':
            header1 = line[7:].strip().split('\t')
            break

    header2 = next(hhmfp).strip().split('\t')
    next(hhmfp)

    hh_seq = []
    profile = []
    for line in hhmfp:
        if line[:2] == '//':
            break
        aa = line[0]
        hh_seq.append(aa)

        freqs = line.split(None, 2)[2].split('\t')[:20]
        features = {h: freq(i) for h, i in zip(header1, freqs)}
        assert len(freqs) == 20

        mid = next(hhmfp)[7:].strip().split('\t')

        features.update({h: freq(i) for h, i in zip(header2, mid)})

        profile.append(features)
        next(hhmfp)

    hh_seq = ''.join(hh_seq)
    seq = seq or hh_seq
    profile = vectorize_profile(profile, seq, hh_seq)

    return {
        'seq': seq,
        'profile': profile,
        'neff': neff,
        'header': header1 + header2,
    }


PROFILE_HEADER = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                  'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                  'M->M', 'M->I', 'M->D', 'I->M', 'I->I',
                  'D->M', 'D->D', 'Neff', 'Neff_I', 'Neff_D') # yapf: disable


AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def vectorize_profile(profile,
                      seq,
                      hh_seq,
                      amino_acids=None,
                      profile_header=None):
    if profile_header is None:
        profile_header = PROFILE_HEADER

    if amino_acids is None:
        amino_acids = AMINO_ACIDS

    seqlen = len(seq)
    aalen = len(amino_acids)
    proflen = len(profile_header)
    profmat = np.zeros((seqlen, aalen + proflen + 1), dtype='float')

    for i, aa in enumerate(seq):
        aa_idx = amino_acids.find(aa)
        if aa_idx > -1:
            profmat[i, aa_idx] = 1.

    if len(profile) == len(seq):
        for i, pos in enumerate(profile):
            for j, key in enumerate(profile_header, aalen):
                profmat[i, j] = pos[key]
    else:
        hh_index = -1
        for i, restype in enumerate(seq):
            if restype != 'X':
                hh_index += 1
                assert restype == hh_seq[hh_index]

            if hh_index >= 0:
                for j, key in enumerate(profile_header, aalen):
                    profmat[i, j] = profile[hh_index][key]

    profmat[:, -1] = 1.

    return profmat
