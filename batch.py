#!/usr/bin/env python3.6

import os
from os.path import join as pjoin
import time
import re
import argparse
import pandas as pd
import numpy as np

from paths import *
import common as c
from target_stats import *
from st_stat import make_st_stat_cache


def further_proc(pair: str, d: dict, verbose: bool) -> None:
    if int(d['committedInsts::0']) < 200*10**6:
        print('Run of {} has not finished, will hide its stats'.format(pair))
        return

    hpt = pair.split('_')[0]
    df = pd.read_csv(st_cache, index_col=0)
    d['ST_IPC'] = str(df.loc['ipc::0'][hpt])

    # compute prediction error
    real_ipc = float(d['ST_IPC'])
    pred_ipc = float(d['HPTpredIPC'])
    d['QoS prediction error'] = (pred_ipc - real_ipc) / real_ipc

    # check slot sanity
    d['slot sanity'] = (float(d['numMissSlots::0']) + float(d['numWaitSlots::0']) + \
         float(d['numBaseSlots::0'])) / (float(d['numCycles']) * 8)

    if verbose:
        c.print_line()
        print(pair, ':')
        c.print_dict(d)

    return d



def main():
    parser = argparse.ArgumentParser(usage='specify stat directory')
    parser.add_argument('-s', '--stat-dir', action='store', required=True,
                        help='gem5 output directory'
                       )
    parser.add_argument('-o', '--output', action='store',
                        help='csv to save results'
                       )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='whether output intermediate result'
                       )
    parser.add_argument('-b', '--error-bound', action='store', type=float,
                        default=0.0,
                        help='Threshold to output an entry'
                       )
    opt = parser.parse_args()

    paths = c.pairs(opt.stat_dir)
    paths = c.stat_filt(paths)
    paths = c.time_filt(paths)
    paths = [pjoin(x, 'stats.txt') for x in paths]
    pairs = c.pairs(opt.stat_dir, return_path=False)

    make_st_stat_cache()

    matrix = {}

    for pair, path in zip(pairs, paths):
        d = c.get_stats(path, brief_targets, re_targets=True)
        matrix[pair] = further_proc(pair, d, opt.verbose)

    df = pd.DataFrame(matrix)

    if opt.output:
        df.to_csv(opt.output, index=True)

    errors = df.loc['QoS prediction error'].values
    print('Mean: {}'.format(np.mean(np.abs(errors))))
    print(errors[np.where(np.abs(errors) > opt.error_bound)])


if __name__ == '__main__':
    main()
