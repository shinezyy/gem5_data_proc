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

def draw_queueing_throughput():
    global gm
    gm.set_cur_ax(gm.ax[0])
    plt.sca(gm.cur_ax)
    baseline_stat_dirs = {
            'Xbar4': c.env.data('xbar4-rand'),
            'Xbar4-2': c.env.data('xbar4-rand'),
            }
    stat_dirs = {
            # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
            'Omega16-OPR': c.env.data('xbar-rand'),
            'Xbar16-OPR': c.env.data('omega-rand'),
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
        baseline_df.sort_index(inplace=True)

        if num_points == 0:
            num_points = len(baseline_df)

        baseline_df.loc['mean'] = baseline_df.iloc[-1]
        baseline_df.loc['mean']['queueingD'] = np.mean(baseline_df['queueingD'])
        data = np.concatenate(
                [baseline_df['queueingD'].values[:-1], [0], baseline_df['queueingD'].values[-1:]])
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
        df.sort_index(inplace=True)
        df.loc['mean'] = df.iloc[-1]
        df.loc['mean']['queueingD'] = np.mean(df['queueingD'])
        print(len(df))
        data = np.concatenate([df['queueingD'].values[:-1], [0],df['queueingD'].values[-1:]])
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
    print(xticklabels)
    legends = [baselines_ordered[0]] + configs_ordered
    fig, ax = gm.reduction_bar_graph(data_all[:2], data_all[2:], xticklabels, legends, 
            # xlabel='Simulation points from SPEC 2017',
            ylabel='Cycles',
            xlim=(-0.5,num_points*num_configs-0.5),
            xtick_scale=1.5,
            title = "(a) Queueing time reduced by M-1 and new interconnect network",
            )


def draw_queueing_rand():
    global gm
    print(gm.ax)
    gm.set_cur_ax(gm.ax[1])
    plt.sca(gm.cur_ax)

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
        baseline_df.sort_index(inplace=True)

        if num_points == 0:
            num_points = len(baseline_df)

        baseline_df.loc['mean'] = baseline_df.iloc[-1]
        baseline_df.loc['mean']['queueingD'] = np.mean(baseline_df['queueingD'])
        data = np.concatenate([baseline_df['queueingD'].values[:-1], [0],baseline_df['queueingD'].values[-1:]])
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
        df.sort_index(inplace=True)
        df.loc['mean'] = df.iloc[-1]
        df.loc['mean']['queueingD'] = np.mean(df['queueingD'])
        data = np.concatenate([df['queueingD'].values[:-1], [0],df['queueingD'].values[-1:]])
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
    for i, benchmark in enumerate(benchmarks_ordered + ["mean"]):
        xticklabels[i*strange_const + 1] = benchmark

    print(num_points, num_configs)
    print(xticklabels)
    legends = baselines_ordered + configs_ordered
    fig, ax = gm.reduction_bar_graph(data_all[:2], data_all[2:], xticklabels, legends, 
            # xlabel='Simulation points from SPEC 2017',
            ylabel='Cycles',
            xlim=(-0.5,num_points*num_configs-0.5),
            xtick_scale=1.5,
            title = "(b) Queueing time reduced by Operand Position Randomization",
            )
    # fig.suptitle('Queueing time reduction', fontsize='large')

def draw_ipc_throughput():
    global gm
    print(gm.ax)
    gm.set_cur_ax(gm.ax[2])
    plt.sca(gm.cur_ax)

    show_reduction = False

    stat_dirs = {
            # 'Xbar4': 'xbar4',
            'Xbar4': 'xbar4-rand',
            # 'Xbar4-SpecSB': 'xbar4-rand-hint',
            # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
            #'Omega16': 'omega',
            'Omega16-OPR': 'omega-rand',
            # 'Omega16-OPR-SpecSB': 'omega-rand-hint',
            #'Xbar16': 'xbar',
            'Xbar16-OPR': 'xbar-rand',
            #'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
            # 'Ideal-OOO': 'ruu-4-issue',
            }
    for k in stat_dirs:
        stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{suffix}')

    configs_ordered = ['Xbar4', 'Omega16-OPR', 'Xbar16-OPR']

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
    dfs['Xbar4'].loc['rel_geo_mean'] = [1.0]
    print('Xbar4')
    print(dfs['Xbar4'])
    for config in configs_ordered:
        if config != 'Xbar4':
            print(config)
            rel = dfs[config]['ipc'] / dfs['Xbar4']['ipc'][:-1]
            dfs[config]['rel'] = rel

            dfs[config].loc['rel_geo_mean'] = [rel.prod() ** (1/len(rel))] * 2

            if config == 'Omega16-OPR-SpecSB':
                dfs[config]['boost'] = dfs[config]['rel'] / dfs['Xbar4']['rel']

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
    print(xticklabels)
    if show_reduction:
        num_configs -= 1
        data_high = np.array([data_all[2], data_all[2]])
        data_low = data_all[:2]
        legends = [configs_ordered[2], configs_ordered[0], configs_ordered[1]]
        fig, ax = gm.reduction_bar_graph(data_high, data_low, xticklabels, legends,
                # xlabel='Simulation points from SPEC 2017',
                # ylabel='IPCs with different configurations',
                ylabel='IPCs',
                xlim=(-0.5, num_points*num_configs),
                ylim=(0, 3))
    else:
        fig, ax = gm.simple_bar_graph(data_all, xticklabels, configs_ordered,
                # xlabel='Simulation points from SPEC 2017',
                ylabel='IPCs',
                # ylabel='IPCs with different configurations',
                xlim=(-0.5, num_points*num_configs-0.5),
                ylim=(0, 3),
                title = "(c) IPC improvements from M-1 and new interconnect network",
                )

draw_queueing_throughput()
draw_queueing_rand()
draw_ipc_throughput()
plt.tight_layout()
gm.save_to_file(plt, 'merged_throughput')
plt.show()

