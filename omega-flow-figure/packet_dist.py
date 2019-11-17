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


stat_dir = c.env.data('xbar-rand-hint')

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

fig, ax = plt.subplots()
ax.set_ylim((0, 1.1))
width = 0.85

colors = ['#7d5c80', '#016201', '#fefe01', 'orange', '#cccccc', '#820000']
cumulative = np.array([0.0] * len(df))
rects = []
for i in range(6):
    if names[i] == 'KeySrcP':
        cumulative = np.array([0.0] * len(df))
    rect = plt.bar(df.index, df[names[i]].values, bottom=cumulative,
            edgecolor='black', color=colors[i], width=width)
    rects.append(rect)
    cumulative += df[names[i]].values

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(np.arange(-0.5, 20 * 3 + 1, 3)))
ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(np.arange(1, 20 * 3 + 1, 3)))

ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
# ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

for tick in ax.xaxis.get_major_ticks():
    # tick.tick1line.set_markersize(0)
    tick.tick2line.set_markersize(0)

for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(0)
    # tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('center')

ax.set_xticklabels(benchmarks_ordered, minor=True, rotation=90)

ax.set_ylabel('Percentage of packet types')
ax.set_xlabel('Simulation points from SPEC 2017')
ax.legend(rects, names, fontsize='small', ncol=6)

plt.tight_layout()
for f in ['eps', 'png']:
    plt.savefig(f'./{f}/packet_targets.{f}', format=f'{f}')

plt.show()
