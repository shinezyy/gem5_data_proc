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

stat_dir = c.env.data('xbar-rand-hint') + suffix

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]
points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')
stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

matrix = {}
for point, stat_file in zip(points, stat_files):
    d = c.get_stats(stat_file, t.packet_targets, re_targets=True)
    matrix[point] = d
df = pd.DataFrame.from_dict(matrix, orient='index')

names = ['TotalP', 'KeySrcP']
data_all = [df['TotalP'].values, df['KeySrcP'].values]
data_all = np.array(data_all)
print(data_all.shape)
benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * (2*len(benchmarks_ordered))
print(len(xticklabels))
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*2] = benchmark

num_configs, num_points = 3, len(benchmarks_ordered)

print(num_points, num_configs)
gm = graphs.GraphMaker(fig_size=(7,4))
gm.config.bar_width, gm.config.bar_interval = 0.7, 0.3
fig, ax = gm.reduction_bar_graph(data_all[:1], data_all[1:], xticklabels, names, 
        xlabel='Simulation points from SPEC 2017',
        ylabel='Number of pointers', 
        xlim=(-0.5, num_points*num_configs-0.5))
# legend = ax.get_legend()
# legend.set_bbox_to_anchor((0.80,0.89))
gm.save_to_file(plt, "crit_pointers")
