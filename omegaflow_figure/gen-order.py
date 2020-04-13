#!/usr/bin/env python3

import os.path as osp
import sys
sys.path.append('..')

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import common as c
import graphs
import target_stats as t


strange_const = 3
show_lins = 62
pd.set_option('precision', 3)
pd.set_option('display.max_rows', show_lins)
pd.set_option('display.min_rows', show_lins)

full = True
do_normalization = True
# do_normalization = False

suffix = '-full' if full else ""
stat_dirs = {
    # 'Ideal-OOO': 'ideal_4w',
    'Realistic-OOO': 'trad_4w',
    'F1': 'f1_rand',
    'O1 w\o WoC non-scale': 'omega-rand',
    'O1 non-scale': 'omega-rand-hint',
    'O1 w\o WoC': 'o1_rand',
    'O1': 'o1_rand_hint',
    'O2 w\o WoC': 'o2_rand',
    'O2': 'o2_rand_hint',
    'O4 w\o WoC': 'o4_rand',
    'O4': 'o4_rand_hint',
}
for k in stat_dirs:
    stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{suffix}')

configs_ordered = [
    # 'Ideal-OOO',
    # 'Realistic-OOO',
    'F1',
    # 'O1 w\o WoC non-scale',
    # 'O1 non-scale',
    # 'O1 w\o WoC',
    # 'O1',
    # 'O2 w\o WoC',
    # 'O2',
    # 'O4 w\o WoC',
    # 'O4',
]

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

num_points = 0
num_configs = len(stat_dirs)
dfs = dict()
benchmarks_ordered = None
baseline = 'Realistic-OOO'
for config in configs_ordered:
    print(config)
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file,
                t.ipc_target + t.packet_targets,
                # t.ipc_target,
                re_targets=True)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')

    if config == 'O1':
        print(df)
        # df['ratioP'] = df['KeySrcP'] / df['TotalP']
        df = df.sort_values(by=['TotalP'])
        benchmarks_ordered = df.index

        with open('./bench_order.txt', 'w') as f:
            for b in benchmarks_ordered:
                f.write(b+'\n')
