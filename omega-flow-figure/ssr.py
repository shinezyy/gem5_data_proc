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


full = True
suffix = '-full' if full else ""

baseline_stat_dir = c.env.data('xbar4') + suffix
stat_dirs = {
        'Xbar4-SpecSB': c.env.data('xbar4-rand-hint'),
        # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
        'Omega16-OPR-SpecSB': c.env.data('omega-rand-hint'),
        # 'Xbar16-OPR-SpecSB': c.env.data('xbar-rand-hint'),
        }
for k in stat_dirs:
    stat_dirs[k] += suffix
configs_ordered = [x for x in stat_dirs]

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

data_all = []
num_points, num_configs = 0, len(stat_dirs)


stat_files = [osp.join(baseline_stat_dir, point, 'stats.txt') for point in points]
matrix = {}
for point, stat_file in zip(points, stat_files):
    d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
    matrix[point] = d
baseline_df = pd.DataFrame.from_dict(matrix, orient='index')
baseline = baseline_df['ssrD'].values

for config in configs_ordered:
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')
    if num_points == 0:
        num_points = len(df)

    print(len(df))

    data_all.append(df['ssrD'].values/baseline)

data_all = np.array(data_all)

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
print(len(xticklabels))
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*2] = benchmark


gm = graphs.GraphMaker()
fig, ax = gm.simple_bar_graph(data_all, xticklabels, configs_ordered, 
        xlabel='Simulation points from SPEC 2017', 
        ylabel='Serialized wakeup delay cycles reduction', 
        xlim=(-0.5, num_points*num_configs-0.5),
        ylim=(0,1))

gm.save_to_file(plt, "ssr")
plt.show()
