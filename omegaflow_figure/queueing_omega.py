#!/usr/bin/env python3

import os.path as osp
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import common as c, target_stats as t
import graphs

strange_const = 3

full = True
suffix = '-full' if full else ""

baseline_stat_dirs = {
        'Xbar4': c.env.data('xbar4-rand'),
        'Xbar4-2': c.env.data('xbar4-rand'),
        }
stat_dirs = {
        # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
        'Omega16-OPR': c.env.data('omega-rand'),
        'Xbar16-OPR': c.env.data('xbar-rand'),
        # 'Xbar16-OPR': c.env.data('xbar-rand'),
        }
for k in baseline_stat_dirs:
    baseline_stat_dirs[k] += suffix
for k in stat_dirs:
    stat_dirs[k] += suffix

baselines_ordered = [x for x in baseline_stat_dirs]
configs_ordered = [x for x in stat_dirs]


benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

num_points = 0
num_configs = len(stat_dirs)

with open('./bench_order.txt') as f:
    index_order = [l.strip() for l in f]

print(index_order)

data_all = []
for baseline in baselines_ordered:
    baseline_stat_dir = baseline_stat_dirs[baseline]
    stat_files = [osp.join(baseline_stat_dir, point, 'stats.txt') for point in points]
    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
        matrix[point] = d
    baseline_df = pd.DataFrame.from_dict(matrix, orient='index')
    baseline_df = baseline_df.reindex(index_order)

    if num_points == 0:
        num_points = len(baseline_df)

    baseline_df.loc['mean'] = baseline_df.iloc[-1]
    baseline_df.loc['mean']['queueingD'] = np.mean(baseline_df['queueingD'])
    data = np.concatenate([baseline_df['queueingD'].values[:-1], [np.NaN], baseline_df['queueingD'].values[-1:]])
    print(data.shape)
    data_all.append(data)

for nc, config in enumerate(configs_ordered):
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
        d['mean'] = np.mean(list(d.values()))
        matrix[point] = d

    df = pd.DataFrame.from_dict(matrix, orient='index')
    df = df.reindex(index_order)

    df.loc['mean'] = df.iloc[-1]
    df.loc['mean']['queueingD'] = np.mean(df['queueingD'])
    print(len(df))
    data = np.concatenate([df['queueingD'].values[:-1], [np.NaN],df['queueingD'].values[-1:]])
    print(data.shape)
    print(data/data_all[nc])
    data_all.append(data)

num_points += 2
data_all = np.array(data_all)
print(data_all.shape)

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
print(len(xticklabels))
for i, benchmark in enumerate(benchmarks_ordered + ["mean"]):
    xticklabels[i*strange_const + 1] = benchmark

print(num_points, num_configs)
gm = graphs.GraphMaker(legend_loc='upper center')
legends = baselines_ordered + configs_ordered
fig, ax = gm.reduction_bar_graph(data_all[:2], data_all[2:], xticklabels, legends,
        # xlabel='Simulation points from SPEC 2017',
        ylabel='Queueing cycles reduction',
        xlim=(-0.5,num_points*num_configs-0.5),
        colors=[['red', 'gray'], ['green', 'green']],
        markers=[['x', '+'], ['.', '.']],
        redundant_baseline=True,
        )
# fig.suptitle('Queueing time reduction', fontsize='large')
# legend = ax.get_legend()
# legend.set_bbox_to_anchor((0.80,0.89))
gm.save_to_file("interconnect_queueing")

plt.show(block=True)
