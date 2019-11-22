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

stat_dirs = {
        # 'Xbar4': 'xbar4',
        'Xbar4': 'xbar4-rand',
        # 'Xbar4-SpecSB': 'xbar4-rand-hint',
        # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
        # 'Omega16': 'omega',
        # 'Omega16-OPR': 'omega-rand',
        # 'Omega16-OPR-SpecSB': 'omega-rand-hint',
        # 'Xbar16': 'xbar',
        # 'Xbar16-OPR': 'xbar-rand',
        # 'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
        # 'Ideal-OOO': 'ruu-4-issue',
        }
for k in stat_dirs:
    stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{suffix}')
configs_ordered = [x for x in stat_dirs]

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

data_all = []
bounds = ['by_bw', 'by_crit_ptr']
num_points, num_configs = 0, len(bounds)

for bound in bounds:
    stat_dir = stat_dirs['Xbar4']
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.standard_targets + t.packet_targets, re_targets=True)
        c.add_packet(d)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')
    if num_points == 0:
        num_points = len(df)

    data_all.append(df[bound].values)

data_all = np.array(data_all)

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
for i, benchmark in enumerate(benchmarks_ordered):
    xticklabels[i*3 + 1] = benchmark

gm = graphs.GraphMaker(fig_size=(6,3))
fig, ax = gm.simple_bar_graph(data_all, xticklabels, bounds,
        xlabel='Simulation points from SPEC 2017',
        ylabel='IPC upper bound with different configs',
        xlim=(-0.5, num_points*num_configs-0.5),
        ylim=(0,4.6))

gm.save_to_file(plt, "upper2")
plt.show(block=True)
