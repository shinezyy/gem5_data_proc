#!/usr/bin/env python3

import os.path as osp
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from utils import common as c, target_stats as t
import graphs

full = True
suffix = '-full' if full else ""

baseline_stat_dirs = {
        'Xbar4': c.env.data('xbar4-rand'),
        'Omega16-OPR': c.env.data('omega-rand'),
        }
stat_dirs = {
        'Xbar4-SpecSB': c.env.data('xbar4-rand-hint'),
        # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
        'Omega16-OPR-SpecSB': c.env.data('omega-rand-hint'),
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

data_all = []
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
    data_all.append(baseline_df['queueingD'].values)

for config in configs_ordered:
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')

    print(len(df))
    data_all.append(df['queueingD'].values)

data_all = np.array(data_all)
print(data_all.shape)

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
print(len(xticklabels))
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*2] = benchmark

print(num_points, num_configs)
gm = graphs.GraphMaker()
legends = baselines_ordered + configs_ordered
fig, ax = gm.reduction_bar_graph(data_all[:2], data_all[2:], xticklabels, legends, 
        xlabel='Simulation points from SPEC 2017',
        ylabel='Relative queueing cycles reduction',
        xlim=(-0.5, num_points*num_configs-0.5))
fig.suptitle('Queueing time reduction', fontsize='large')
# legend = ax.get_legend()
# legend.set_bbox_to_anchor((0.80,0.89))
gm.save_to_file("spec_sb_queueing")
