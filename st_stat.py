#!/usr/bin/env python3.6

import os
from os.path import join as pjoin
from os.path import expanduser as expu
import numpy as np
import pandas as pd

import common as c
from paths import st_stat_dir


def regenerate_st_stat(pairs: list, cache: str) -> None:
    st_targets = [
        'system\.cpu\.(ipc::0)',
    ]
    matrix = {}
    for p in pairs:
        matrix[p] = c.get_stats(pjoin(st_stat_dir, p, 'stats.txt'),
                              st_targets, re_targets=True)
    df = pd.DataFrame(matrix)
    df.to_csv(cache, index=True)

    df = pd.read_csv(cache, index_col=0)
    print(df)


def make_st_stat_cache() -> None:
    pairs = c.pairs(st_stat_dir, False)
    stat_files = [pjoin(st_stat_dir, x, 'stats.txt') \
                  for x in pairs]

    stat_timestamps = [os.path.getmtime(x) for x in stat_files]
    newest = max(stat_timestamps)
    script = os.path.realpath(__file__)
    cache_file = pjoin(os.path.dirname(script), 'cache_st_stat.txt')

    if not os.path.isfile(cache_file) or \
       newest > os.path.getmtime(cache_file) or \
       os.path.getmtime(script) > os.path.getmtime(cache_file):
        regenerate_st_stat(pairs, cache_file)


if __name__ == '__main__':
    make_st_stat_cache()

