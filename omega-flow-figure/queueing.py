#!/usr/bin/env python3

import os.path as osp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys
sys.path.append('.')

import common as c
import graphs
import target_stats as t

full = True
suffix = '-full' if full else ""

show_reduction = False

baseline_stat_dir = c.env.data('xbar4-rand') + suffix
stat_dirs = {
        'Xbar4-SpecSB': c.env.data('xbar4-rand-hint'),
        # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
        'Omega16-OPR': c.env.data('omega-rand'),
        'Omega16-OPR-SpecSB': c.env.data('omega-rand-hint'),
        # 'Xbar16-OPR': c.env.data('xbar-rand'),
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
baseline = baseline_df['queueingD'].values

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
    data_all.append(1 - df['queueingD'].values / baseline)
    print(1 - df['queueingD'].values/baseline)

data_all = np.array(data_all)

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
print(len(xticklabels))
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*2] = benchmark

print(data_all.shape)
gm = graphs.GraphMaker()
if show_reduction:
    data_high = np.array([data_all[1], data_all[3]])
    data_low = np.array([data_all[0], data_all[2]])
    legends = [configs_ordered[1], configs_ordered[3], configs_ordered[0], configs_ordered[2]]
    num_configs -= 2
    fig, ax = gm.reduction_bar_graph(data_high, data_low, xticklabels, legends, 
            xlabel='Simulation points from SPEC 2017', 
            ylabel='Relative queueing cycles reduction', 
            xlim=(-0.5, num_points*num_configs-0.5), 
            ylim=(0, 3), legendorder=(2,0,3,1))
else:
    fig, ax = gm.simple_bar_graph(data_all, xticklabels, configs_ordered, 
            xlabel='Simulation points from SPEC 2017', 
            ylabel='Relative queueing cycles reduction', 
            xlim=(-0.5, num_points*num_configs-0.5),
            ylim=(0,1.05))
    fig.suptitle('Queueing time reduction', fontsize='large')
gm.save_to_file(plt, "queueing")
plt.show()

