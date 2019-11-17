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
import target_stats as t


baseline_stat_dirs = {
        'Omega16': c.env.data('omega'),
        'Xbar16': c.env.data('xbar'),
        }
stat_dirs = {
        # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
        'Omega16-OPR': c.env.data('omega-rand'),
        'Xbar16-OPR': c.env.data('xbar-rand'),
        # 'Xbar16-OPR': c.env.data('xbar-rand'),
        }

baselines_ordered = [x for x in baseline_stat_dirs]
configs_ordered = [x for x in stat_dirs]

colors = ['black', 'r', '#820000', '#00c100', 'white', '#fefe01', 'black', '#604080']
baseline_colors = ['black', 'r']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

fig, ax = plt.subplots()
fig.set_size_inches(8, 4, forward=True)
width = 0.9
interval = 0.1

rects = []

shift = 0.0
num_points = 0
num_configs = len(stat_dirs)


i = 0
for baseline in baselines_ordered:
    baseline_stat_dir = baseline_stat_dirs[baseline]
    stat_files = [osp.join(baseline_stat_dir, point, 'stats.txt') for point in points]
    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
        matrix[point] = d
    baseline_df = pd.DataFrame.from_dict(matrix, orient='index')

    if num_points == 0:
        num_points = len(baseline_df)

    tick_starts = np.arange(0, num_points * num_configs + 2, (width + interval) * num_configs) + shift
    print(len(tick_starts))
    print(tick_starts)
    baseline_df.loc['mean'] = baseline_df.iloc[-1]
    baseline_df.loc['mean']['queueingD'] = np.mean(baseline_df['queueingD'])
    print(len(baseline_df))
    print(baseline_df)
    rect = plt.bar(tick_starts,
            baseline_df['queueingD'].values, color='white',
            edgecolor=baseline_colors[i],
            width=width)
    rects.append(rect)
    shift += width + interval
    i += 1

i = 0
shift = 0.0
for config in configs_ordered:
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
        d['mean'] = np.mean(list(d.values()))
        matrix[point] = d

    df = pd.DataFrame.from_dict(matrix, orient='index')
    df.loc['mean'] = df.iloc[-1]
    df.loc['mean']['queueingD'] = np.mean(df['queueingD'])

    print(len(df))

    tick_starts = np.arange(0, num_points * num_configs + 2, (width + interval) * num_configs) + shift
    print(tick_starts)
    rect = plt.bar(tick_starts,
        df['queueingD'].values,
        # edgecolor='black',
        color=colors[i], width=width)
    rects.append(rect)
    shift += width + interval
    i += 1

ax.set_xlim((-0.6, num_points * num_configs + 2))
# ax.set_ylim((0, 1.1))

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
xticklabels.append('mean')

ax.set_xticklabels(xticklabels, minor=True, rotation=90)

ax.set_ylabel('Relative queueing cycles reduction')
ax.set_xlabel('Simulation points from SPEC 2017')
ax.legend(rects, baselines_ordered + configs_ordered, fontsize='small', ncol=5)

fig.suptitle('Queueing time reduction', fontsize='large')

plt.tight_layout()
for f in ['eps', 'png']:
    plt.savefig(f'./{f}/rand_queueing.{f}', format=f'{f}')

plt.show()
