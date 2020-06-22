#!/usr/bin/env python3

import os
import sys
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

pd.set_option('precision', 3)


def further_proc(pair: str, d: dict, verbose: bool) -> None:
    hpt, lpt = pair.split('_')
    # c.add_st_ipc(hpt, d)
    # c.add_overall_qos(hpt, lpt, d)
    # c.add_ipc_pred(d)
    # c.add_slot_sanity(d)
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
    parser.add_argument('--branch', action='store_true')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='whether output intermediate result'
                       )
    parser.add_argument('-b', '--error-bound', action='store', type=float,
                        default=0.0,
                        help='Threshold to output an entry'
                       )
    parser.add_argument('-i', '--ipc-only', action='store_true',
                        default=0.0,
                        help='Only extract ipc'
                       )
    parser.add_argument('--st', action='store_true',
                        help='processing ST stats'
                       )
    parser.add_argument('--pair-filter', action='store', default='',
                        help='file than filt pairs'
                       )
    parser.add_argument('-f', '--stat-file', action='store',
                        help='name of stats file', default='stats.txt'
                       )
    parser.add_argument('-l', '--fanout', action='store_true',
                        help='print fanout'
                       )
    parser.add_argument('--fetch', action='store_true',
                        help='print fetch info'
                       )
    opt = parser.parse_args()

    pairs = c.pairs(opt.stat_dir, return_path=False)

    if (opt.pair_filter):
        pairs = c.pair_filt(pairs, opt.pair_filter)

    paths = c.pair_to_full_path(opt.stat_dir, pairs)

    pairs, paths = c.stat_filt(pairs, paths, opt.stat_file)
    # paths = c.time_filt(paths)
    paths = [pjoin(x, opt.stat_file) for x in paths]

    # make_st_stat_cache()

    matrix = {}

    for pair, path in zip(pairs, paths):
        print(pair)
        if opt.ipc_only:
            d = c.get_stats(path, ipc_target, re_targets=True)
        else:
            targets = brief_targets
            if opt.branch:
                targets += branch_targets
            if opt.fanout:
                targets += fanout_targets
            if opt.fetch:
                targets += fetch_targets

            d = c.get_stats(path, targets, re_targets=True)

        if len(d):
            if not opt.st:
                matrix[pair] = further_proc(pair, d, opt.verbose)
            else:
                matrix[pair] = d
            if opt.branch:
                c.add_branch_mispred(d)
            if opt.fanout:
                c.add_fanout(d)

    df = pd.DataFrame.from_dict(matrix, orient='index')
    df = df.sort_index(1)
    df = df.sort_values(['ipc'])
    for x in df.index:
        print(x)
    if len(df):
        df.loc['mean'] = df.mean()

    df = df.fillna(0)

    print(df)

    if opt.output:
        df.to_csv(opt.output, index=True)

    # print('filted QoS')
    # print(df['QoS_0'][df['QoS_0'] < 0.9])

if __name__ == '__main__':
    main()
