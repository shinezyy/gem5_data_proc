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
import target_stats as t

prefix = '../'
full = False
if full:
    suffix = '-full'
else:
    suffix = ''
stat_dirs = {
        # 'Xbar4': 'xbar4',
        'Xbar4': 'xbar4-rand',
        'Xbar4-SpecSB': 'xbar4-rand-hint',
        # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
        #'Omega16': 'omega',
        'Omega16-OPR': 'omega-rand',
        'Omega16-OPR-SpecSB': 'omega-rand-hint',
        #'Xbar16': 'xbar',
        #'Xbar16-OPR': 'xbar-rand',
        #'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
        #'Ideal-OOO': 'ruu-4-issue',
        }
for k in stat_dirs:
    stat_dirs[k] = osp.join(prefix, f'{stat_dirs[k]}{suffix}')

configs_ordered = [x for x in stat_dirs]

colors = ['#454545', '#820000', '#00c100', 'orange', '#7d5c80', 'black',
        'pink', '#fefe01', 'orange']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

fig, ax = plt.subplots()
fig.set_size_inches(10, 5, forward=True)
width = 0.6
interval = 0.4

rects = []

shift = 0.0
i = 0
num_points = 0
num_configs = len(stat_dirs)
for config in configs_ordered:
    print(config)
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.ipc_target, re_targets=True)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')
    if num_points == 0:
        num_points = len(df)

    # print(len(df))

    tick_starts = np.arange(0, num_points * num_configs, (width + interval) * num_configs) + shift

    print(df['ipc'].values[:10])
    # print(tick_starts)
    rect = plt.bar(tick_starts,
        df['ipc'].values,
        # edgecolor='black',
        color=colors[i], width=width)
    rects.append(rect)
    shift += width + interval
    i += 1

ax.set_xlim((-0.6, num_points * num_configs))
ax.set_ylim((0, 3))

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(
    np.arange(-0.5, (num_points + 1) * num_configs, (width + interval) * num_configs * 3)))

ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(
    np.arange(-0.5, (num_points + 1) * num_configs, (width + interval) * num_configs)))

ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
# # ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
#
for tick in ax.xaxis.get_major_ticks():
    tick.tick1line.set_markersize(10)
    tick.tick2line.set_markersize(0)

for tick in ax.xaxis.get_minor_ticks():
    tick.tick1line.set_markersize(2)
    # tick.tick2line.set_markersize(0)
    tick.label1.set_horizontalalignment('left')

xticklabels = [''] * num_points
# print(len(xticklabels))
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*3 + 1] = benchmark

ax.set_xticklabels(xticklabels, minor=True, rotation=90)

ax.set_ylabel('IPCs with different configurations')
ax.set_xlabel('Simulation points from SPEC 2017')
ax.legend(rects, configs_ordered, fontsize='small', ncol=num_configs)

plt.tight_layout()
for f in ['eps', 'png']:
    plt.savefig(f'./{f}/ipc.{f}', format=f'{f}')

plt.show()

