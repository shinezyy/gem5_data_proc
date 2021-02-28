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

full = '_f'
bp = '_o'
# bp = '_perceptron'
lbuf = '_lbuf'

suffix = f'_ltu1_b2b_x5{lbuf}{bp}{full}'

n_cols = 1
n_rows = 3

gm = graphs.GraphMaker(
        fig_size=(7, 5.5),
        multi_ax=True,
        legend_loc='best',
        with_xtick_label=False,
        frameon=False,
        nrows=n_rows,
        ncols=n_cols,
        sharex='all')

with open('./bench_order.txt') as f:
    index_order = [l.strip() for l in f]

colors = plt.get_cmap('Dark2').colors

def spec_on_xbar4():
    global gm
    gm.set_cur_ax(gm.ax[0])
    plt.sca(gm.cur_ax)

    baseline_stat_dirs = {
            'F1': c.env.data('f1_rand'),
            # 'Omega16-OPR': c.env.data('omega-rand'),
            }
    stat_dirs = {
            # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
            'F1 w/ WoC': c.env.data('xbar4-rand-hint'),
            # 'Omega16-OPR-SpecSB': c.env.data('omega-rand-hint'),
            # 'Xbar16-OPR': c.env.data('xbar-rand'),
            }
    stats = ['queueingDelay', 'ssrDelay']
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
        baseline_df = baseline_df.reindex(index_order)

        if num_points == 0:
            num_points = len(baseline_df)

        baseline_df.loc['mean'] = baseline_df.iloc[-1]
        baseline_df.loc['mean'][stat] = np.mean(baseline_df[stat])

        data = np.concatenate([baseline_df[stat].values[:-1], [np.NaN],
            baseline_df[stat].values[-1:]])
        # print(data.shape)
        data_all.append(data)

    legends = [
            'F1 Queuing', 'F1 SSR',
            'F1 w/ WoC Queuing', 'F1 w/ WoC SSR',
            ]

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
        df = df.reindex(index_order)

        df.loc['mean'] = df.iloc[-1]
        df.loc['mean'][stat] = np.mean(df[stat])
        data = np.concatenate([df[stat].values[:-1],
            [np.NaN],df[stat].values[-1:]])
        # print(data/data_all[nc])
        # print(data.shape)
        data_all.append(data)


    num_points += 2
    data_all = np.array(data_all)
    # print(data_all.shape)

    benchmarks_ordered = []
    for point in df.index:
        if point.endswith('_0'):
            benchmarks_ordered.append(point.split('_')[0])

    xticklabels = [''] * num_points
    # print(len(xticklabels))
    for i, benchmark in enumerate(benchmarks_ordered + ['mean']):
        xticklabels[i*strange_const + 1] = benchmark

    # print(num_points, num_configs)
    legends = [
            'F1 Queuing', 'F1 SSR',
            'F1 w/ WoC Queuing', 'F1 w/ WoC SSR',
            ]
    fig, ax = gm.simple_bar_graph(data_all, xticklabels, legends,
            ylabel='Cycles',
            xlim=(-0.5, num_points-0.5),
            ylim=(0,1.22e9),
            title='(a) Effect of WoC on Forwardflow',
            colors=[colors[0], colors[2], colors[1], colors[3], ],
            markers=['+', 7, 'x', 6],
            dont_legend=True,
            )

    ax.legend(
            legends,
            loc = 'upper left',
            # bbox_to_anchor=(0, 0),
            ncol=2,
            fancybox=True,
            framealpha=0.5,
            fontsize=13,
            )


def spec_on_omega():
    global gm
    gm.set_cur_ax(gm.ax[0])
    plt.sca(gm.cur_ax)

    baseline_stat_dirs = {
            # 'Xbar4': c.env.data('xbar4-rand'),
            'O1 w/o WoC': c.env.data('o1_rand'),
            }
    stat_dirs = {
            # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
            # 'Xbar4-SpecSB': c.env.data('xbar4-rand-hint'),
            'O1': c.env.data('o1_rand_hint'),
            # 'Xbar16-OPR': c.env.data('xbar-rand'),
            }
    stats = ['queueingDelay', 'ssrDelay']
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
    m200 = 200 * 10**6

    for stat in stats:
        baseline_stat_dir = baseline_stat_dirs[baselines_ordered[0]]
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
        baseline_df.loc['mean'][stat] = np.mean(baseline_df[stat])

        data = np.concatenate([baseline_df[stat].values[:-1] / m200, [np.NaN],
            baseline_df[stat].values[-1:] / m200])
        # print(data.shape)
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
        df = df.reindex(index_order)

        df.loc['mean'] = df.iloc[-1]
        df.loc['mean'][stat] = np.mean(df[stat])
        data = np.concatenate([df[stat].values[:-1] / m200,
            [np.NaN], df[stat].values[-1:] / m200])
        # print(data/data_all[nc])
        # print(data.shape)
        data_all.append(data)

    num_points += 2
    data_all = np.array(data_all)
    # print(data_all.shape)

    benchmarks_ordered = []
    for point in df.index:
        if point.endswith('_0'):
            benchmarks_ordered.append(point.split('_')[0])

    xticklabels = [''] * num_points
    # print(len(xticklabels))
    for i, benchmark in enumerate(benchmarks_ordered + ['mean']):
        xticklabels[i*strange_const + 1] = benchmark

    # print(num_points, num_configs)

    # data_all order:
    # 'Omega16 Queueing', 'Omega16 SSR',
    # 'Omega16-SpecSB Queueing', 'Omega16-SpecSB SSR',

    improve = data_all[2][-1] / data_all[0][-1] - 1
    print(f'Token queueing cycle reduction of WoC: {improve}')

    improve = data_all[3][-1] / data_all[1][-1] - 1
    print(f'SSR delayed cycle reduction of WoC: {improve}')

    legends = ['O1 w/o WoC', 'O1']
    fig, ax = gm.simple_bar_graph(
            np.array([data_all[0], data_all[2]]),
            xticklabels,
            legends=legends,
            ylabel='Token queuing cycles\nper instruction',
            xlim=(-0.5, num_points-0.5),
            title='(a) Impact of WoC on token queueing cycles',
            # colors=['red', 'gray'],
            colors=[colors[0], colors[1]],
            markers=[7, 6],
            dont_legend=True,
            )

    ax.legend(
            legends,
            loc = 'upper left',
            # bbox_to_anchor=(0, 0),
            ncol=2,
            fancybox=True,
            framealpha=0.5,
            fontsize=13,
            )

    gm.set_cur_ax(gm.ax[1])
    plt.sca(gm.cur_ax)
    fig, ax = gm.simple_bar_graph(
            np.array([data_all[1], data_all[3]]),
            xticklabels,
            legends=legends,
            ylabel='SSR delayed cycles\nper instruction',
            xlim=(-0.5, num_points-0.5),
            title='(b) Impact of WoC on SSR delayed cycles',
            # colors=['red', 'gray'],
            colors=[colors[0], colors[1]],
            markers=[7, 6],
            dont_legend=True,
            )

    ax.legend(
            legends,
            loc = 'upper left',
            # bbox_to_anchor=(0, 0),
            ncol=2,
            fancybox=True,
            framealpha=0.5,
            fontsize=13,
            )


def ipc_spec():
    global gm
    gm.set_cur_ax(gm.ax[2])
    plt.sca(gm.cur_ax)

    show_reduction = True

    stat_dirs = {
            # 'F1': 'f1',
            # 'F1 w/ WoC': 'xbar4-rand-hint',
            # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
            #'Omega16': 'omega',
            'O1 w/o WoC': 'o1_rand',
            'O1': 'o1_rand_hint',
            #'Xbar16': 'xbar',
            #'Xbar16-OPR': 'xbar-rand',
            #'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
            # 'Ideal-OOO': 'ruu-4-issue',
            }
    for k in stat_dirs:
        stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{suffix}')
    stat_dirs['ideal'] = c.env.data(f'trad_4w{lbuf}{bp}{full}')
    # print(stat_dirs)

    # configs_ordered = ['Xbar4', 'Xbar4-SpecSB','Omega16-OPR', 'Omega16-OPR-SpecSB']
    configs_ordered = [
        # 'F1', 'F1 w/ WoC',
        'O1 w/o WoC', 'O1', 'ideal']

    benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

    points = []
    for b in benchmarks:
        for i in range(0, 3):
            points.append(f'{b}_{i}')

    data_all = []
    num_points, num_configs = 0, len(stat_dirs)
    dfs = dict()

    for config in configs_ordered:
        stat_dir = osp.expanduser(stat_dirs[config])
        stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

        matrix = {}
        for point, stat_file in zip(points, stat_files):
            d = c.get_stats(stat_file, t.ipc_target, re_targets=True)
            matrix[point] = d

        df = pd.DataFrame.from_dict(matrix, orient='index')
        df = df.reindex(index_order)

        dfs[config] = df

        if num_points == 0:
            num_points = len(df)

    baseline = 'ideal'
    dfs[baseline].loc['rel_geo_mean'] = [1.0]
    # print(baseline)
    # print(dfs[baseline])
    datas = dict()
    for config in configs_ordered:
        if config != baseline:
            # print('Shape of', config, dfs[config]['ipc'].shape)
            rel = dfs[config]['ipc'] / dfs[baseline]['ipc'][:-1]
            dfs[config]['rel'] = rel

            rel_geo_mean = rel.prod() ** (1/len(rel))

            # if config in ['O1', 'O1 w/o WoC']:
            #     print(dfs[config])
            #     print(dfs[baseline])
                # dfs[config]['boost'] = dfs[config]['rel'] / dfs[baseline]['rel']

            # print(dfs[config])
            data = np.concatenate([dfs[config]['rel'].values,
                [np.NaN], [rel_geo_mean]])
            datas[config] = data

    improve = datas['O1'][-1] - datas['O1 w/o WoC'][-1]
    print(f'IPC improvement of WoC: {improve}')

    # legends = ['Omega16-OPR-SpecSB', 'Xbar4-SpecSB', 'Omega16-OPR', 'Xbar4']
    legends = ['O1 w/o WoC', 'O1', ]
    data_all = [datas[x] for x in legends]
    # print(data_all)
    data_all = np.array(data_all)
    # print('Shape of data all:', data_all.shape)

    num_points += 2

    benchmarks_ordered = []
    for point in df.index:
        if point.endswith('_0'):
            benchmarks_ordered.append(point.split('_')[0])

    xticklabels = [''] * num_points
    for i, benchmark in enumerate(benchmarks_ordered + ['rel_geomean']):
        xticklabels[i*strange_const + 1] = benchmark

    fig, ax = gm.simple_bar_graph(
        data_all, xticklabels,
        legends,
        ylabel='Normalized IPC',
        xlabel='Simulation points from SPECCPU 2017 (sorted by TPI)',
        xlim=(-0.5, num_points-0.5),
        # ylim=(0, 3),
        title='(c) Impact of WoC on IPC',
        # colors=['red', 'gray', 'green', 'blue'],
        colors=[colors[0], colors[1], colors[0], colors[2], ],
        markers=[6, 7],
        with_borders=False,
        dont_legend=True,
    )
    ax.legend(
            legends,
            loc = 'best',
            # bbox_to_anchor=(0, 0),
            ncol=2,
            fancybox=True,
            framealpha=0.5,
            fontsize=13,
            )


# spec_on_xbar4()

spec_on_omega()
ipc_spec()

plt.tight_layout()

gm.save_to_file("spec_merge")
# plt.show(block=True)

