import common as c
import target_stats as t
import os
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

top_dir = '/home51/zyy/projects/gem5_data_proc/expri_results/gem5_ooo_spec06_shotgun/gcc_200/'


def get_ipc(stat_path: str):
    targets = t.ipc_target
    stats = c.get_stats(stat_path, targets, insts=100*10**6, re_targets=True)
    return stats['ipc']


def glob_stats(path: str):
    stat_files = []
    for x in os.listdir(path):
        stat_path = osp.join(path, x, 'm5out/stats.txt')
        if osp.isfile(stat_path) and osp.getsize(stat_path) > 10 * 1024:  # > 10K
            stat_files.append((x, stat_path))
    return stat_files


def gen_csv(workload, name, input_dir):
    d = {}
    for i, f in glob_stats(input_dir):
        d[int(i)] = get_ipc(f)

    print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.sort_index(ascending=True)
    os.makedirs(f"outputs/{workload}")
    df.to_csv(f"outputs/{workload}/{name}.csv")


def draw(csv):
    df = pd.read_csv(csv, index_col=[0])
    df2 = {}
    with open('outputs/class_map.txt') as f:
        for line in f:
            k, v = line.strip().split(' ')
            df2[int(k)] = df.iloc[int(v)][0]

    df2 = pd.DataFrame.from_dict(df2, orient='index')
    df2 = df2.sort_index(ascending=True)

    print(df.mean())
    print(df2.mean())

    df = df.loc[400: 500]
    df2 = df2.loc[400: 500]

    print(df)
    print(df2)

    fig = plt.gcf()
    fig.set_size_inches(16, 5)

    plt.step(df.index, df.values)
    plt.step(df2.index, df2.values)

    plt.tight_layout()
    plt.show()


workload = 'gcc_200'
name = 'shotgun_ipc'
draw(f"outputs/{workload}/{name}.csv")