#!/usr/bin/env python3

import os.path as osp
import sys
sys.path.append('.')

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import common as c
import graphs
import target_stats as t


show_lins = 62
pd.set_option('precision', 3)
pd.set_option('display.max_rows', show_lins)
pd.set_option('display.min_rows', show_lins)

full = True
suffix = '-full' if full else ""

stat_dirs = {
        # 'Xbar4': 'xbar4',
        'Xbar4': 'xbar4-rand',
        # 'Xbar4-SpecSB': 'xbar4-rand-hint',
        # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
        #'Omega16': 'omega',
        #'Omega16-OPR': 'omega-rand',
        'Omega16-OPR-SpecSB': 'omega-rand-hint',
        #'Xbar16': 'xbar',
        #'Xbar16-OPR': 'xbar-rand',
        #'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
        'Ideal-OOO': 'ruu-4-issue',
        }
for k in stat_dirs:
    stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{suffix}')

configs_ordered = ['Xbar4', 'Omega16-OPR-SpecSB', 'Ideal-OOO']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

num_points = 0
dfs = dict()
for config in configs_ordered:
    print(config)
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
        matrix[point] = d

    df = pd.DataFrame.from_dict(matrix, orient='index')
    dfs[config] = df

    if num_points == 0:
        num_points = len(df)

targets = ['ssrD', 'queueingD']
num_targets = len(targets)
data_all = []
for target in targets:
    df = dfs[configs_ordered[0]]
    print(len(df[target].values))
    print(df[target].values)
    data_all.append(df[target].values)

data_all = np.array(data_all)

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*2] = benchmark

gm = graphs.GraphMaker()
legends = ['Delay by SSR', 'Delay by Queueing']
fig, ax = gm.simple_bar_graph(data_all, xticklabels, legends, 
        xlabel='Simulation points from SPEC 2017', ylabel='!!!!!!!!WRONG LABEL', 
        xlim=(-0.5, num_points*num_targets-0.5))

gm.save_to_file(plt, "ssr_queueing")
plt.show()
