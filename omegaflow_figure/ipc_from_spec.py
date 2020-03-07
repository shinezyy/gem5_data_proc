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

strange_const = 3

show_lins = 62
pd.set_option('precision', 3)
pd.set_option('display.max_rows', show_lins)
pd.set_option('display.min_rows', show_lins)

full = True
suffix = '-full' if full else ""

show_reduction = True

stat_dirs = {
        'Xbar4': 'xbar4',
        'Xbar4-SpecSB': 'xbar4-rand-hint',
        # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
        #'Omega16': 'omega',
        'Omega16-OPR': 'omega-rand',
        'Omega16-OPR-SpecSB': 'omega-rand-hint',
        #'Xbar16': 'xbar',
        #'Xbar16-OPR': 'xbar-rand',
        #'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
        # 'Ideal-OOO': 'ruu-4-issue',
        }
for k in stat_dirs:
    stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{suffix}')

configs_ordered = ['Xbar4', 'Xbar4-SpecSB','Omega16-OPR', 'Omega16-OPR-SpecSB']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

data_all = []
num_points, num_configs = 0, len(stat_dirs)
dfs = dict()
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

    dfs[config] = df

    if num_points == 0:
        num_points = len(df)

baseline = 'Xbar4'
dfs[baseline].loc['rel_geo_mean'] = [1.0]
print(baseline)
print(dfs[baseline])
for config in configs_ordered:
    if config != baseline:
        print(config)
        rel = dfs[config]['ipc'] / dfs[baseline]['ipc'][:-1]
        dfs[config]['rel'] = rel

        dfs[config].loc['rel_geo_mean'] = [rel.prod() ** (1/len(rel))] * 2

        if config.endswith('SpecSB'):
            print(dfs[config])
            print(dfs[baseline])
            # dfs[config]['boost'] = dfs[config]['rel'] / dfs[baseline]['rel']

        print(dfs[config])
    data = np.concatenate([dfs[config]['ipc'].values[:-1], [0], dfs[config]['ipc'].values[-1:]])
    data_all.append(data)
num_points += 2
data_all = np.array(data_all)

benchmarks_ordered = []
for point in df.index:
    if point.endswith('_0'):
        benchmarks_ordered.append(point.split('_')[0])

xticklabels = [''] * num_points
for i, benchmark in enumerate(benchmarks_ordered + ['rel_geomean']):
    xticklabels[i*strange_const + 1] = benchmark

print(data_all.shape)
gm = graphs.GraphMaker()
if show_reduction:
    data_high = np.array([data_all[1], data_all[3]])
    data_low = np.array([data_all[0], data_all[2]])
    legends = [configs_ordered[1], configs_ordered[3], configs_ordered[0], configs_ordered[2]]
    num_configs -= 2
    fig, ax = gm.reduction_bar_graph(data_high, data_low, xticklabels, legends, 
            xlabel='Simulation points from SPEC 2017', 
            ylabel='IPCs with different configurations', 
            xlim=(-0.5, num_points*num_configs), 
            ylim=(0, 3), legendorder=(2,0,3,1))
else:
    fig, ax = gm.simple_bar_graph(data_all, xticklabels, configs_ordered, 
            xlabel='Simulation points from SPEC 2017', 
            ylabel='IPCs with different configurations', 
            xlim=(-0.5, num_points*num_configs-0.5), 
            ylim=(0, 3))
gm.save_to_file("ipc_from_spec")
plt.show(block=True)
