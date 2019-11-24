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

full = True
suffix = '-full' if full else ""

n_cols = 1
n_rows = 3

gm = graphs.GraphMaker((14,7), True, n_rows, n_cols, sharex='all')

def spec_on_xbar4():
    global gm
    gm.set_cur_ax(gm.ax[0])
    plt.sca(gm.cur_ax)

    baseline_stat_dirs = {
            'Xbar4': c.env.data('xbar4-rand'),
            # 'Omega16-OPR': c.env.data('omega-rand'),
            }
    stat_dirs = {
            # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
            'Xbar4-SpecSB': c.env.data('xbar4-rand-hint'),
            # 'Omega16-OPR-SpecSB': c.env.data('omega-rand-hint'),
            # 'Xbar16-OPR': c.env.data('xbar-rand'),
            }
    stats = ['queueingD', 'ssrD']
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

    num_points, num_configs = 0, len(stat_dirs)

    data_all = []
    for stat in stats:
        baseline_stat_dir = baseline_stat_dirs[baselines_ordered[0]]
        stat_files = [osp.join(baseline_stat_dir, point, 'stats.txt') for point in points]
        matrix = {}
        for point, stat_file in zip(points, stat_files):
            d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
            matrix[point] = d
        baseline_df = pd.DataFrame.from_dict(matrix, orient='index')
        baseline_df.sort_index(inplace=True)

        if num_points == 0:
            num_points = len(baseline_df)

        baseline_df.loc['mean'] = baseline_df.iloc[-1]
        baseline_df.loc['mean'][stat] = np.mean(baseline_df[stat])

        data = np.concatenate([baseline_df[stat].values[:-1], [0],
            baseline_df[stat].values[-1:]])
        print(data.shape)
        data_all.append(data)

    for nc, stat in enumerate(stats):
        stat_dir = stat_dirs[configs_ordered[0]]
        stat_dir = osp.expanduser(stat_dir)
        stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

        matrix = {}
        for point, stat_file in zip(points, stat_files):
            d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
            d['mean'] = np.mean(list(d.values()))
            matrix[point] = d

        df = pd.DataFrame.from_dict(matrix, orient='index')
        df.sort_index(inplace=True)
        df.loc['mean'] = df.iloc[-1]
        df.loc['mean'][stat] = np.mean(df[stat])
        data = np.concatenate([df[stat].values[:-1], [0],df[stat].values[-1:]])
        print(data/data_all[nc])
        print(data.shape)
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
    for i, benchmark in enumerate(benchmarks_ordered + ['mean']):
        xticklabels[i*strange_const + 1] = benchmark

    print(num_points, num_configs)
    legends = [
            'Xbar4 Queueing Cycles', 'Xbar4 SSR Delay Cycles',
            'Xbar4-SpecSB Queueing Cycles', 'Xbar4-SpecSB SSR Delay Cycles',
            ]
    fig, ax = gm.reduction_bar_graph(data_all[:2], data_all[2:], xticklabels, legends,
            ylabel='Cycles',
            xlim=(-0.5, num_points*2-0.5),
            ylim=(0,1.22e9),
            title='(a) Effect of Speculative Scoreboard/ARF  on Xbar4',
            )

def spec_on_omega():
    global gm
    gm.set_cur_ax(gm.ax[1])
    plt.sca(gm.cur_ax)

    baseline_stat_dirs = {
            # 'Xbar4': c.env.data('xbar4-rand'),
            'Omega16-OPR': c.env.data('omega-rand'),
            }
    stat_dirs = {
            # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
            # 'Xbar4-SpecSB': c.env.data('xbar4-rand-hint'),
            'Omega16-OPR-SpecSB': c.env.data('omega-rand-hint'),
            # 'Xbar16-OPR': c.env.data('xbar-rand'),
            }
    stats = ['queueingD', 'ssrD']
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

    num_points, num_configs = 0, len(stat_dirs)

    data_all = []

    for stat in stats:
        baseline_stat_dir = baseline_stat_dirs[baselines_ordered[0]]
        stat_files = [osp.join(baseline_stat_dir, point, 'stats.txt') for point in points]
        matrix = {}
        for point, stat_file in zip(points, stat_files):
            d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
            matrix[point] = d
        baseline_df = pd.DataFrame.from_dict(matrix, orient='index')
        baseline_df.sort_index(inplace=True)

        if num_points == 0:
            num_points = len(baseline_df)

        baseline_df.loc['mean'] = baseline_df.iloc[-1]
        baseline_df.loc['mean'][stat] = np.mean(baseline_df[stat])

        data = np.concatenate([baseline_df[stat].values[:-1], [0],
            baseline_df[stat].values[-1:]])
        print(data.shape)
        data_all.append(data)

    for nc, stat in enumerate(stats):
        stat_dir = stat_dirs[configs_ordered[0]]
        stat_dir = osp.expanduser(stat_dir)
        stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

        matrix = {}
        for point, stat_file in zip(points, stat_files):
            d = c.get_stats(stat_file, t.breakdown_targets, re_targets=True)
            d['mean'] = np.mean(list(d.values()))
            matrix[point] = d

        df = pd.DataFrame.from_dict(matrix, orient='index')
        df.sort_index(inplace=True)

        df.loc['mean'] = df.iloc[-1]
        df.loc['mean'][stat] = np.mean(df[stat])
        data = np.concatenate([df[stat].values[:-1], [0],df[stat].values[-1:]])
        print(data/data_all[nc])
        print(data.shape)
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
    for i, benchmark in enumerate(benchmarks_ordered + ['mean']):
        xticklabels[i*strange_const + 1] = benchmark

    print(num_points, num_configs)
    legends = [
            'Omega16 Queueing Cycles', 'Omega16 SSR Delay Cycles',
            'Omega16-SpecSB Queueing Cycles', 'Omega16-SpecSB SSR Delay Cycles',
            ]
    fig, ax = gm.reduction_bar_graph(data_all[:2], data_all[2:], xticklabels, legends, 
            ylabel='Cycles',
            xlim=(-0.5, num_points*2-0.5),
            title='(b) Effect of Speculative Scoreboard/ARF  on Omega16',
            )

def ipc_spec():
    global gm
    gm.set_cur_ax(gm.ax[2])
    plt.sca(gm.cur_ax)

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
        df.sort_index(inplace=True)

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
    if show_reduction:
        data_high = np.array([data_all[1], data_all[3]])
        data_low = np.array([data_all[0], data_all[2]])
        legends = [configs_ordered[1], configs_ordered[3], configs_ordered[0], configs_ordered[2]]
        num_configs -= 2
        fig, ax = gm.reduction_bar_graph(data_high, data_low, xticklabels, legends, 
                ylabel='IPCs',
                xlim=(-0.5, num_points*num_configs), 
                ylim=(0, 3), legendorder=(2,0,3,1),
                title='(c) IPC improvements from Speculative Scoreboard/ARF on Omega16 and Xbar4',
                )
    else:
        fig, ax = gm.simple_bar_graph(data_all, xticklabels, configs_ordered, 
                ylabel='IPCs',
                xlim=(-0.5, num_points*num_configs-0.5), 
                ylim=(0, 3),
                title='(c) IPC improvements from Speculative Scoreboard/ARF on Omega16 and Xbar4',
                )

spec_on_xbar4()
spec_on_omega()
ipc_spec()

plt.tight_layout()

gm.save_to_file(plt, "spec_merge")
plt.show()

