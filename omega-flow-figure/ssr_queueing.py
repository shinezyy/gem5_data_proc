#!/usr/bin/env python3

import os.path as osp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys
sys.path.append('..')

import common as c
import target_stats as t

show_lins = 62
pd.set_option('precision', 3)
pd.set_option('display.max_rows', show_lins)
pd.set_option('display.min_rows', show_lins)

prefix = '~/gem5-results-2017/'
full = False
if full:
    suffix = '-full'
else:
    suffix = ''
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
    stat_dirs[k] = osp.join(prefix, f'{stat_dirs[k]}{suffix}')

configs_ordered = ['Xbar4', 'Omega16-OPR-SpecSB', 'Ideal-OOO']

colors = ['#454545', '#820000', '#00c100', 'orange', '#7d5c80', 'black',
        'pink', '#fefe01', 'orange']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

fig, ax = plt.subplots()
fig.set_size_inches(14, 4, forward=True)
width = 0.6
interval = 0.4

rects = []

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

shift = 0.0
i = 0
targets = ['ssrD', 'queueingD']
num_targets = len(targets)
for target in targets:
    df = dfs[configs_ordered[0]]
    # print(len(df))
    tick_starts = np.arange(0, num_points * num_targets, (width + interval) * num_targets) + shift
    # print(tick_starts)
    print(len(tick_starts))
    print(tick_starts)
    print(len(df[target].values))
    print(df[target].values)
    rect = plt.bar(tick_starts,
        df[target].values,
        # edgecolor='black',
        color=colors[i], width=width)
    rects.append(rect)
    shift += width + interval
    i += 1

ax.set_xlim((-0.6, num_points * num_targets))
# ax.set_ylim((0, 3))

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(
    np.arange(-0.5, (num_points + 1) * num_targets, (width + interval) * num_targets * 3)))

ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(
    np.arange(-0.5, (num_points + 1) * num_targets, (width + interval) * num_targets)))

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
# xticklabels.append('rel_geomean')

ax.set_xticklabels(xticklabels, minor=True, rotation=90)

ax.set_ylabel('IPCs with different configurations')
ax.set_xlabel('Simulation points from SPEC 2017')
ax.legend(rects, ['Delay by SSR', 'Delay by Queueing'], fontsize='small', ncol=num_targets)

plt.tight_layout()
for f in ['eps', 'png']:
    plt.savefig(f'./{f}/ssr_queueing.{f}', format=f'{f}')

plt.show()

