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
do_normalization = True
# do_normalization = False
colors = plt.get_cmap('Dark2').colors

with open('./bench_order.txt') as f:
    index_order = [l.strip() for l in f]

full = '_f'
bp = '_o'
# bp = '_perceptron'
lbuf = '_lbuf'

suffix = f'_ltu1_b2b_x5{lbuf}{bp}{full}'
ooo_suffix = f'{lbuf}{bp}{full}'

n_cols = 1
n_rows = 2

gm = graphs.GraphMaker(
        fig_size=(7, 3.5),
        multi_ax=True,
        legend_loc='best',
        with_xtick_label=False,
        frameon=False,
        nrows=n_rows,
        ncols=n_cols,
        sharex='all')


def get_stat_dirs():
    return {
        'Idealized-OoO': f'trad_4w{ooo_suffix}',
        'Realistic-OoO': f'realistic_trad_4w{ooo_suffix}',
        'F1': f'f1_rand{suffix}',
        'O1*': f'o1_rand{suffix}',
        'O1': f'o1_rand_hint{suffix}',
        'O2*': f'o2_rand{suffix}',
        'O2': f'o2_rand_hint{suffix}',
        'O4*': f'o4_rand{suffix}',
        'O4': f'o4_rand_hint{suffix}',
    }


def ipc_scale():
    global gm
    gm.set_cur_ax(gm.ax[0])
    plt.sca(gm.cur_ax)

    stat_dirs = get_stat_dirs()
    for k in stat_dirs:
        stat_dirs[k] = c.env.data(f'{stat_dirs[k]}')

    baseline = 'Realistic-OoO'
    # configs_ordered = ['Xbar4', 'Xbar4-SpecSB','Omega16-OPR', 'Omega16-OPR-SpecSB']
    configs_ordered = [baseline, 'O4', 'O2', 'O1', ]

    benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

    points = []
    for b in benchmarks:
        for i in range(0, 3):
            points.append(f'{b}_{i}')

    data_all = []
    num_points, num_configs = 0, len(stat_dirs)
    dfs = dict()
    for config in configs_ordered:
        # print(config)
        stat_dir = stat_dirs[config]
        stat_dir = osp.expanduser(stat_dir)
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

    dfs[baseline].loc['rel_geo_mean'] = [1.0]
    # print(baseline)
    # print(dfs[baseline])
    datas = dict()
    for config in configs_ordered:
        if config != baseline:
            # print(config)
            rel = dfs[config]['ipc'] / dfs[baseline]['ipc'][:-1]
            dfs[config]['rel'] = rel

            dfs[config].loc['rel_geo_mean'] = [rel.prod() ** (1 / len(rel))] * 2

            # print(dfs[config])
            stat = 'rel'
            data = np.concatenate([dfs[config][stat].values[:-1],
                                   [np.NaN], dfs[config][stat].values[-1:]])
            print(f'IPC improvement of {config}: {(dfs[config][stat].values[-1:][0] - 1.0)*100}%', )
            datas[config] = data

    # legends = ['Omega16-OPR-SpecSB', 'Xbar4-SpecSB', 'Omega16-OPR', 'Xbar4']
    legends = configs_ordered[1:]

    data_all = [datas[x] for x in legends]
    # print(data_all)
    data_all = np.array(data_all)
    # print(data_all.shape)

    num_points += 2

    benchmarks_ordered = []
    for point in df.index:
        if point.endswith('_0'):
            benchmarks_ordered.append(point.split('_')[0])

    xticklabels = [''] * num_points
    for i, benchmark in enumerate(benchmarks_ordered + ['rel_geomean']):
        xticklabels[i * strange_const + 1] = benchmark

    # print(data_all.shape)
    fig, ax = gm.simple_bar_graph(data_all, xticklabels,
                                  legends,
                                  ylabel='Normalized IPC',
                                  xlim=(-0.5, num_points - 0.5),
                                  ylim=(0.5, 1.6),
                                  title='(a) IPC improvements from scaling up',
                                  # colors=['red', 'gray', 'green', 'blue'],
                                  colors=[colors[1], colors[2], colors[0],],
                                  markers=['+', 'x', 6],
                                  with_borders=False,
                                  dont_legend=True,
                                  )
    ax.legend(
        legends,
        loc='upper left',
        # bbox_to_anchor=(0, 0),
        ncol=2,
        fancybox=True,
        framealpha=0.5,
        fontsize=13,
    )


def scale_vs_woc():
    global gm
    gm.set_cur_ax(gm.ax[1])
    plt.sca(gm.cur_ax)

    stat_dirs = get_stat_dirs()
    for k in stat_dirs:
        stat_dirs[k] = c.env.data(f'{stat_dirs[k]}')

    # configs_ordered = ['Xbar4', 'Xbar4-SpecSB','Omega16-OPR', 'Omega16-OPR-SpecSB']
    baseline = 'Realistic-OoO'
    configs_ordered = [baseline, 'O2', 'O2*', 'O4*', ]

    benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

    points = []
    for b in benchmarks:
        for i in range(0, 3):
            points.append(f'{b}_{i}')

    data_all = []
    num_points, num_configs = 0, len(stat_dirs)
    dfs = dict()
    for config in configs_ordered:
        # print(config)
        stat_dir = stat_dirs[config]
        stat_dir = osp.expanduser(stat_dir)
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

    dfs[baseline].loc['rel_geo_mean'] = [1.0]
    # print(baseline)
    # print(dfs[baseline])
    datas = dict()
    for config in configs_ordered:
        if config != baseline:
            # print(config)
            rel = dfs[config]['ipc'] / dfs[baseline]['ipc'][:-1]
            dfs[config]['rel'] = rel

            dfs[config].loc['rel_geo_mean'] = [rel.prod() ** (1 / len(rel))] * 2

            # print(dfs[config])
            stat = 'rel'
            data = np.concatenate([dfs[config][stat].values[:-1],
                                   [np.NaN], dfs[config][stat].values[-1:]])
            print(f'IPC improvement of {config}: {(dfs[config][stat].values[-1:][0] - 1.0)*100}%', )
            datas[config] = data

    # legends = ['Omega16-OPR-SpecSB', 'Xbar4-SpecSB', 'Omega16-OPR', 'Xbar4']
    legends = configs_ordered[1:]

    data_all = [datas[x] for x in legends]
    # print(data_all)
    data_all = np.array(data_all)
    # print(data_all.shape)

    num_points += 2

    benchmarks_ordered = []
    for point in df.index:
        if point.endswith('_0'):
            benchmarks_ordered.append(point.split('_')[0])

    xticklabels = [''] * num_points
    for i, benchmark in enumerate(benchmarks_ordered + ['rel_geomean']):
        xticklabels[i * strange_const + 1] = benchmark

    # print(data_all.shape)
    fig, ax = gm.simple_bar_graph(data_all, xticklabels,
                                  legends,
                                  ylabel='Normalized IPC',
                                  xlabel='Simulation points from SPECCPU 2017 (sorted by TPI)',
                                  xlim=(-0.5, num_points - 0.5),
                                  ylim=(0.5, 1.6),
                                  title='(b) WoC VS. scaling up',
                                  # colors=['red', 'gray', 'green', 'blue'],
                                  colors=[colors[1], colors[2], colors[0],],
                                  markers=[7, '+', 'x', ],
                                  with_borders=False,
                                  dont_legend=True,
                                  )
    ax.legend(
        legends,
        loc='upper left',
        # bbox_to_anchor=(0, 0),
        ncol=2,
        fancybox=True,
        framealpha=0.5,
        fontsize=13,
    )


ipc_scale()
scale_vs_woc()

plt.tight_layout()

gm.save_to_file("scale_ipc")
# plt.show(block=True)

