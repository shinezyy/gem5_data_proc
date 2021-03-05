import os
import os.path as osp
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


def gen_ipc_seq_from_classmap(map_file, label):
    if isinstance(label, str):
        assert osp.isfile(label)
        label = pd.read_csv(label, index_col=0, header=0)
    else:
        assert isinstance(label, pd.DataFrame)
    class_map = pd.read_csv(map_file, index_col=0, header=None)
    print(class_map)
    predictions = label.loc[class_map[1]]
    predictions.index = class_map.index
    print(predictions)
    return predictions


def step_df(df, base, label,):
    assert base is not None

    index = set(df.index) & set(base.index)
    base = base.loc[index].sort_index()
    df = df.loc[index].sort_index()
    # cumulative_error = (df.values - base.values)/base.values
    print(df.shape)
    print(base.shape)
    cumulative_error = df.values - base.values
    cumulative_base = np.sum(base)
    print(cumulative_error)
    print(f'cumulative error: {sum(np.abs(cumulative_error.reshape(-1)))/cumulative_base}')
    print(f'IPC error: {(df.values.mean() - base.values.mean())/base.values.mean()}')

    start = (50*10**6) * 0
    l = (50*10**6) * 4000

    print(int(start + l))

    df = df.loc[start: int(start + l)].reset_index(drop=True)
    print(len(df))
    base = base.loc[start: int(start + l)].reset_index(drop=True)
    print(len(base))
    plt.step(df.index, df.values, label=label[0])
    plt.step(base.index, base.values, label=label[1])
    plt.fill_between(
        df.index, df.values.reshape(-1), base.values.reshape(-1),
        step='pre',
    )


def draw_betapoint(truth, pred):
    truth = pd.read_csv(truth, index_col=[0])['ipc']
    if not isinstance(pred, pd.DataFrame):
        assert isinstance(pred, str)
        print(pred)
        # assert osp.isfile(pred)
        pred = pd.read_csv(pred, index_col=[0])
    # pred = pred[pred['Similarity'] > 0.0]['IPC']
    if 'IPC' in pred.columns:
        pred = pred['IPC']
    elif 'ipc' in pred.columns:
        pred = pred['ipc']
    else:
        pred = pred['0']
    dfs = [truth, pred]
    labels = ['Supervised(BetaPoint)', 'Ground truth', ]

    fig = plt.gcf()
    fig.set_size_inches(64, 20)

    base = None
    step_df(pred, truth, labels)

    plt.legend()
    plt.tight_layout()
    plt.show()


def main(): 
    workload = 'gcc_200'
    # workload = 'gcc_200-pure'
    # workload = 'bzip2_liberty'
    name = 'shotgun_ipc'
    # draw_xgb(f"outputs/{workload}/{name}.csv")

    # pred = gen_ipc_seq_from_classmap(
    #     '/home51/zyy/projects/betapoint/models/simpoint_class_maps/simpoint_class_map_dr_3_k15.txt',
    #     f'outputs/shotgun_continuous_point/{workload}_smarts8w.csv',
    # )
    
    # pred = f'~/predictions/beta_pred/{workload}-sim960-99.csv'
    pred = f'~/predictions/beta_pred/{workload}-smarts_sim450.csv'
    # pred = f'~/predictions/beta_pred/{workload}-pure.csv'
    # pred = '~/predictions/beta_pred/gcc_200_Lgbp.csv'
    truth = f'outputs/shotgun_continuous_point/{workload}_smarts8w.csv'
    # truth = 'outputs/shotgun_continuous_point/gcc_200_sampled.csv'

    # draw_betapoint(truth, pred)
    draw_betapoint(truth, pred)


if __name__ == '__main__':
    main()
