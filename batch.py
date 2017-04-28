#!/usr/bin/env python3.6

import os
from os.path import join as pjoin
import time
import re
import argparse
from pprint import pprint
import pandas as pd

from paths import *
import common as c
from target_stats import *
from st_stat import make_st_stat_cache


def print_stat(pair: str, d: dict) -> None:
    c.print_line()
    if int(d['committedInsts::0']) < 200*10**6:
        print('Run of {} has not finished, will hide its stats'.format(pair))
        return

    hpt = pair.split('_')[0]
    df = pd.read_csv(st_cache, index_col=0)
    d['st_IPC'] = str(df.loc['ipc::0'][hpt])
    real_ipc = float(d['st_IPC'])
    pred_ipc = float(d['HPTpredIPC'])
    d['QoS prediction error'] = '{:.6f}'.format((pred_ipc - real_ipc) / real_ipc)

    print(pair, ':')
    c.print_dict(d)



def main():
    parser = argparse.ArgumentParser(usage='specify stat directory')
    parser.add_argument('-s', '--stat-dir', action='store', required=True,
                        help='gem5 output directory'
                       )
    opt = parser.parse_args()

    paths = c.pairs(opt.stat_dir)
    paths = c.stat_filt(paths)
    paths = c.time_filt(paths)
    paths = [pjoin(x, 'stats.txt') for x in paths]
    pairs = c.pairs(opt.stat_dir, return_path=False)

    make_st_stat_cache()

    for pair, path in zip(pairs, paths):
        d = c.get_stats(path, brief_targets, re_targets=True)
        print_stat(pair, d)

if __name__ == '__main__':
    main()
