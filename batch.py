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
    hpt, lpt = pair.split('_')
    # c.add_st_ipc(hpt, d)
    c.add_overall_qos(hpt, lpt, d)
    c.add_ipc_pred(d)
    c.add_slot_sanity(d)
    # c.add_qos(d)

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
    parser.add_argument('--st', action='store_true',
                        help='processing ST stats'
                       )
    opt = parser.parse_args()

    paths = c.pairs(opt.stat_dir)
    paths = c.stat_filt(paths)
    # paths = c.time_filt(paths)
    paths = [pjoin(x, 'stats.txt') for x in paths]
    pairs = c.pairs(opt.stat_dir, return_path=False)

    make_st_stat_cache()

    matrix = {}

    for pair, path in zip(pairs, paths):
        d = c.get_stats(path, brief_targets, re_targets=True)
        if len(d):
            if not opt.st:
                matrix[pair] = further_proc(pair, d, opt.verbose)
            else:
                matrix[pair] = d

    df = pd.DataFrame.from_dict(matrix, orient='index')

    if not opt.st:
        errors = df['IPC prediction error'].values
        print('Mean: {}'.format(np.mean(np.abs(errors))))
        # df.sort_values(['QoS prediction error'], ascending=False, inplace=True)

        print(df['overall QoS'][abs(df['IPC prediction error']) > opt.error_bound])

    if opt.output:
        df.to_csv(opt.output, index=True)

    print('filted QoS')
    print(df['QoS_0'][df['QoS_0'] < 0.9])

if __name__ == '__main__':
    main()
