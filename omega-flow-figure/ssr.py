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

baseline_stat_dir = '../xbar4'
stat_dirs = {
        'Xbar4-SpecSB': '../xbar4-rand-hint',
        # 'Xbar4*2-SpecSB': '../dedi-xbar4-rand-hint',
        'Omega16-OPR-SpecSB': '../omega-rand-hint',
        # 'Xbar16-OPR-SpecSB': '../xbar-rand-hint',
        }
configs_ordered = [x for x in stat_dirs]

colors = ['#454545', '#fefe01', '#820000', '#00c100', '#7d5c80', 'black']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

fig, ax = plt.subplots()
fig.set_size_inches(10, 5, forward=True)
width = 0.8
interval = 0.2

rects = []

shift = 0.0
i = 0
num_points = 0
num_configs = len(stat_dirs)


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

    tick_starts = np.arange(0, num_points * num_configs, (width + interval) * num_configs) + shift
    print(tick_starts)
    plt.bar(tick_starts,
        [1.0] * num_points, edgecolor='grey',
        color='white', width=width)
    rect = plt.bar(tick_starts,
        df['ssrD'].values/baseline, edgecolor='black',
        color=colors[i], width=width)
    rects.append(rect)
    shift += width + interval
    i += 1

ax.set_xlim((-0.6, num_points * num_configs))
ax.set_ylim((0, 1.0))

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
print(len(xticklabels))
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*3 + 1] = benchmark

ax.set_xticklabels(xticklabels, minor=True, rotation=90)

ax.set_ylabel('Serialized wakeup delay cycles reduction')
ax.set_xlabel('Simulation points from SPEC 2017')
ax.legend(rects, configs_ordered, fontsize='small', ncol=5)

plt.tight_layout()
for f in ['eps', 'png']:
    plt.savefig(f'./{f}/ssr.{f}', format=f'{f}')

plt.show()

