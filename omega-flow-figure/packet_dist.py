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

for i in df.columns.values:
    if i != 'TotalP':
        df[i] = df[i] / df['TotalP']
df.drop('TotalP', axis=1, inplace=True)
names = df.columns.values.tolist()
names.remove('DestOpP')
names.append('DestOpP')
names.remove('KeySrcP')
names.append('KeySrcP')
print(type(names))
print(names)
df = df.reindex(columns=names)
print(df)

gm = graphs.GraphMaker(fig_size=(7,4))
gm.config.bar_width, gm.config.bar_interval = 0.7, 0.3

cumulative = np.array([0.0] * len(df))
rects = []
data_all = []
for i in range(6):
    if names[i] == 'KeySrcP':
        cumulative = np.array([0.0] * len(df))
    rect = plt.bar(df.index, df[names[i]].values, bottom=cumulative,
            edgecolor=gm.config.edgecolor, color=gm.config.colors[i], width=gm.config.bar_width)
    rects.append(rect)
    data_all.append(df[names[i]].values)
    cumulative += df[names[i]].values

data_all = np.array(data_all)
print(data_all.shape)

num_points = data_all.shape[1]
benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
for i, benchmark in enumerate(benchmarks_ordered + ['rel_geomean']):
    xticklabels[i*2] = benchmark

bar_size = gm.config.bar_width + gm.config.bar_interval
gm.ax.xaxis.set_major_locator(mpl.ticker.IndexLocator(base=bar_size*3, offset=-gm.config.bar_interval/2))
gm.ax.xaxis.set_minor_locator(mpl.ticker.IndexLocator(base=bar_size, offset=-gm.config.bar_interval/2))

gm.set_graph_general_format(xlim=(-0.5, num_points-0.5), 
        ylim=(0, 1), xticklabels=xticklabels, 
        xlabel='Simulation points from SPEC 2017',
        ylabel='Percentage of packet types')
gm.ax.legend(rects, names, fontsize='small', ncol=6)

# plt.tight_layout()

gm.save_to_file("packet_targets")
plt.show(block=True)
