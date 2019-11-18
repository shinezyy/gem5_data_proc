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

stat_dirs = {
        # 'Xbar4': 'xbar4',
        'Xbar4': 'xbar4-rand',
        # 'Xbar4-SpecSB': 'xbar4-rand-hint',
        # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
        # 'Omega16': 'omega',
        'Omega16-OPR': 'omega-rand',
        # 'Omega16-OPR-SpecSB': 'omega-rand-hint',
        # 'Xbar16': 'xbar',
        'Xbar16-OPR': 'xbar-rand',
        # 'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
        # 'Ideal-OOO': 'ruu-4-issue',
        }
for k in stat_dirs:
    stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{suffix}')

configs_ordered = [x for x in stat_dirs]

colors = ['white', '#454545', '#fefe01', '#820000', '#00c100', '#7d5c80', 'black',
        'pink', 'orange']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = ['xz_0', 'namd_2', 'imagick_1', 'lbm_0']
# for b in benchmarks:
#     for i in range(0, 3):
#         points.append(f'{b}_{i}')

n_cols = 2
n_rows = 2

fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all')
fig.set_size_inches(8, 5, forward=True)

width = 0.8
interval = 0.2

rects = []

num_x = 0
num_configs = len(stat_dirs)
df = None

count = 0
for point in points:
    shift = 0.0
    sub_ax = axs[count//n_cols][count%n_cols]
    plt.axes(sub_ax)
    for j, config in enumerate(configs_ordered):
        print(config)
        stat_dir = stat_dirs[config]
        stat_dir = osp.expanduser(stat_dir)
        stat_file = osp.join(stat_dir, point, 'stats.txt')
        print(stat_dir)

        matrix = {}
        d = c.get_stats(stat_file, t.flow_target, re_targets=True)
        print(d)
        d.pop('WKFlowUsage::total')
        d.pop('WKFlowUsage::0')

        # cycle to packets
        for k in d:
            n = int(k.split('::')[1]) * d[k]
            d[k] = n
        matrix[point] = d

        df = pd.DataFrame.from_dict(matrix, orient='index')
        if num_x == 0:
            num_x = len(df.columns)

        # print(len(df))

        tick_starts = np.arange(0, num_x * num_configs, (width + interval) * num_configs) + shift
        # print(tick_starts)
        rect = plt.bar(tick_starts,
            df.iloc[0].values, edgecolor='black',
            color=colors[j], width=width)
        rects.append(rect)
        shift += width + interval

        sub_ax.set_xlim((-0.6, num_x * num_configs))
        # ax.set_ylim((0, 3))

        benchmarks_ordered = []
        for point in df.index:
            if point.endswith('_0'):
                benchmarks_ordered.append(point.split('_')[0])

        sub_ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(
            np.arange(-0.5, (num_x + 1) * num_configs, (width + interval) * num_configs)))

        sub_ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(
            np.arange(-2, (num_x + 1) * num_configs, (width + interval) * num_configs)))

        sub_ax.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
        # sub_ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        for tick in sub_ax.xaxis.get_major_ticks():
            tick.tick1line.set_markersize(3)
            tick.tick2line.set_markersize(0)

        for tick in sub_ax.xaxis.get_minor_ticks():
            tick.tick1line.set_markersize(0)
            tick.tick2line.set_markersize(0)
            tick.label1.set_horizontalalignment('left')

        xticklabels = [''] * num_x * num_configs
        for i, x in enumerate(df.columns):
            xticklabels[i + 1] = x.split('::')[1]

        sub_ax.set_xticklabels(xticklabels, minor=True)
        sub_ax.set_title(point, fontsize='small')
        if count == 1:
            sub_ax.legend(rects, configs_ordered, fontsize='small', ncol=1)
        if count == 2:
            sub_ax.set_ylabel('Packet consumed', ha='left')
            sub_ax.set_xlabel('Packet consumed per cycle', ha='left')
    count += 1

fig.suptitle("Packet consuming distribution", fontsize=14)


plt.tight_layout()
for f in ['eps', 'png']:
    plt.savefig(f'./{f}/flow_dist.{f}', format=f'{f}')

plt.show()
