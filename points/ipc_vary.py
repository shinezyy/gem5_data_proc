import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import glob_stats, get_ipc
from utils.points_extraction import class_map_to_ipc_df

top_dir = '/home51/zyy/projects/gem5_data_proc/expri_results/gem5_ooo_spec06_shotgun/gcc_200/'


def gen_csv(workload, name, input_dir):
    d = {}
    for i, f in glob_stats(input_dir):
        d[int(i)] = get_ipc(f)

    print(d)
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.sort_index(ascending=True)
    os.makedirs(f"outputs/{workload}")
    df.to_csv(f"outputs/{workload}/{name}.csv")


def step_df(df, label, base):
    if base is not None:
        index = set(df.index) & set(base.index)
        base = base.loc[index]
        df = df.loc[index]
        cumulative_error = (df.values - base.values)/base.values
        print(cumulative_error)
        print(f'cumulative error: {sum(np.abs(cumulative_error.reshape(-1))/len(df.index))}')
        print(f'IPC error: {(df.values.mean() - base.values.mean())/base.values.mean()}')

    df = df.loc[1000:1200]
    plt.step(df.index, df.values, label=label)
    if base is not None:
        base = base.loc[1000:1200]
        plt.fill_between(
            df.index, df.values.reshape(-1), base.values.reshape(-1),
            step='pre',
        )


def draw(csv):
    df = pd.read_csv(csv, index_col=[0])
    df = df.sort_index(ascending=True)
    dfs = [df]
    labels = ['Ground truth']
    print(df.mean())

    # df2 = class_map_to_ipc_df(df, "/home51/zyy/projects/betapoint/models/simpoint_class_map.txt")
    # dfs.append(df2)
    # labels.append('Simpoint')
    # print(df2.mean())

    # df3 = class_map_to_ipc_df(df, 'outputs/class_map_min_2.txt')
    df3 = class_map_to_ipc_df(df, 'outputs/class_map_min.txt')
    dfs.append(df3)
    labels.append('Supervised(XGBoost)')
    print(df3.mean())

    fig = plt.gcf()
    fig.set_size_inches(16, 5)

    base = None
    for df_, label in zip(dfs, labels):
        step_df(df_, label, base)
        if base is None:
            base = df_

    plt.legend()
    plt.tight_layout()
    plt.show()


workload = 'gcc_200'
name = 'shotgun_ipc'
draw(f"outputs/{workload}/{name}.csv")