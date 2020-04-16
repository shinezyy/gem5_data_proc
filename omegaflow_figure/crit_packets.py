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

colors = plt.get_cmap('Dark2').colors

strange_const = 3
do_normalization = True
full = True
suffix = '-full' if full else ""

stat_dir = c.env.data('o1_rand_hint') + suffix

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

names = ['critical', 'non-critical']
if not do_normalization:
    data_all = [df['TotalP'].values, df['KeySrcP'].values]
else:
    data_all = [df['TotalP'].values / df['TotalP'].values,
            df['KeySrcP'].values / df['TotalP'].values]
data_all = np.array(data_all)
print(data_all.shape)
benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

# xticklabels = [''] * (strange_const*len(benchmarks_ordered))
# print(len(xticklabels))
# for i, benchmark in enumerate(benchmarks_ordered):
#     xticklabels[i*strange_const + 1] = benchmark
xticklabels = benchmarks_ordered

num_configs, num_points = 3, len(benchmarks_ordered)

print(num_points, num_configs)
gm = graphs.GraphMaker(fig_size=(6,2.5))
gm.config.bar_width, gm.config.bar_interval = 0.7, 0.3
common_options = (data_all[:1], data_all[1:], xticklabels, names)
print(data_all[1])

non_crit_min = 1-np.max(data_all[1])
non_crit_max = 1-np.min(data_all[1])
print('non-crit min:', non_crit_min,
      'non-crit max:', non_crit_max)

speed_up = 3
min_ = 1/((1-non_crit_min) + non_crit_min/speed_up)
max_ = 1/((1-non_crit_max) + non_crit_max/speed_up)
print(f'Speed up by {min_}~{max_}')


if not do_normalization:
    fig, ax = gm.reduction_bar_graph(*common_options,
            ylabel='Number of tokens',
            xlim=(-0.5, num_points*num_configs-0.5),
            )
else:
    fig, ax = gm.reduction_bar_graph(
        *common_options,
        colors=[[colors[0]], [colors[1]]],
        ylabel='Proportion of tokens',
        # xlim=(-0.5, num_points*num_configs-0.5),
        # ylim=(0.0, 1.0),
        bar=True,
        show_tick=True,
        swap=True,
        show_minor=True,
    )

# legend.set_bbox_to_anchor((0.80,0.89))
plt.tight_layout()
gm.save_to_file("crit_tokens")
# plt.show(block=True)
