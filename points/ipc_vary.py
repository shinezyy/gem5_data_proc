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


def gen_seq_from_classmap(map_file, label, target='CPI'):
    if isinstance(label, str):
        assert osp.isfile(label)
        label = pd.read_csv(label, index_col=0, header=0)
    else:
        assert isinstance(label, pd.DataFrame)
    class_map = pd.read_csv(map_file, index_col=0, header=None)
    print(class_map)
    predicted = []
    for i, row in class_map.iterrows():
        if row[1] in label.index:
            predicted.append(i)
    # predicted = sorted(list(set(class_map.index) & set(label.index)))
    class_map = class_map.loc[predicted]
    predictions = label.loc[class_map[1]]
    predictions.index = class_map.index
    print(predictions)
    return predictions, class_map[1]


def step_df(df, base, label, ylabel='CPI'):
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
    # print(f'Cumulative absolute error: {sum(np.abs(cumulative_error.reshape(-1)))/cumulative_base:5.3f}')
    # print(f'CPI error: {(df.values.mean() - base.values.mean())/base.values.mean():5.3f}')
    print(f'Cumulative absolute error: {sum(np.abs(cumulative_error.reshape(-1)))/cumulative_base}')
    print(f'{ylabel} error: {(df.values.mean() - base.values.mean())/base.values.mean()}')

    start = (50*10**6) * 3000
    l = (50*10**6) * 500

    print(start, start + l)

    df = df.loc[start: int(start + l)]
    print(len(df))
    base = base.loc[start: int(start + l)]
    print(len(base))
    plt.step(df.index, df.values, label=label[0])
    plt.step(base.index, base.values, label=label[1])
    plt.fill_between(
        df.index, df.values.reshape(-1), base.values.reshape(-1),
        step='pre',
        color='yellow',
    )
    plt.ylim((0, 4.0))
    plt.ylabel(ylabel, fontsize=22)
    plt.xlabel('Dynamic instruction count', fontsize=22)


def draw_betapoint(truth, pred, name, target='ipc'):
    fig = plt.gcf()
    fig.set_size_inches(28, 10)

    truth = pd.read_csv(truth, index_col=[0])[target]
    if not isinstance(pred, pd.DataFrame):
        assert isinstance(pred, str)
        # assert osp.isfile(pred)
        pred = pd.read_csv(pred, index_col=[0])
    # pred = pred[pred['Similarity'] > 0.0]['IPC']
    if target == 'ipc':
        if 'IPC' in pred.columns:
            pred = pred['IPC']
        elif 'ipc' in pred.columns:
            pred = pred['ipc']
        else:
            pred = pred['0']
        labels = [name, 'Ground truth CPI gathered from GEM5 O3', ]
        step_df(1.0 / pred, 1.0 / truth, labels)
    elif target == 'L2MPKI':
        print('Drawing L2 MPKI')
        pred = pred['L2MPKI']
        labels = [name, 'Ground truth MPKI gathered from GEM5 O3', ]
        step_df(pred, truth, labels, ylabel='L2MPKI')

    elif target == 'condIncorrect':
        print('Drawing Branch MPKI')
        pred = pred['condIncorrect']
        labels = [name, 'Ground truth Branch MPKI gathered from GEM5 O3', ]
        step_df(pred, truth, labels, ylabel='BranchMPKI')

    else:
        assert False

    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.show()


def main(): 
    workload = 'gcc_200'
    # workload = 'gcc_200-pure'
    # workload = 'bzip2_liberty'
    name = 'shotgun_ipc'
    # draw_xgb(f"outputs/{workload}/{name}.csv")

    # pred = f'~/predictions/beta_pred/{workload}-smarts_sim450.csv'
    # pred = f'~/predictions/beta_pred/{workload}-smarts_sim1300_full82_99.csv'

    # pred, known = gen_seq_from_classmap(
    #     '/home51/zyy/projects/betapoint/models/simpoint_class_maps/simpoint_class_map_dr_4_k300.txt',
    #     f'outputs/shotgun_continuous_point/{workload}_smarts8w_mpki.csv',
    # )
    # known = known.values

    # known = pd.read_csv('./outputs/shotgun_continuous_point/gcc_200_sampled_3.csv', index_col=[0])

    pred = f'~/predictions/beta_pred/{workload}-smarts8w_large-mae0.03x.csv'
    pred = pd.read_csv(pred, index_col=[0])
    known = pd.read_csv('./outputs/shotgun_continuous_point/gcc_200_smarts8w_sampled.csv', index_col=[0])
    known = known.index

    print('Length of known:', len(set(known)))
    unknown = sorted(list(set(pred.index) - set(known)))
    pred = pred.loc[unknown]

    # pred = f'~/predictions/beta_pred/{workload}-sim960-99.csv'
    # pred = f'~/predictions/beta_pred/{workload}-pure.csv'
    # pred = '~/predictions/beta_pred/gcc_200_Lgbp.csv'
    truth = f'outputs/shotgun_continuous_point/{workload}_smarts8w_mpki.csv'
    # draw_betapoint(truth, pred, 'Our supervised method')
    draw_betapoint(truth, pred,
                   # 'SimPoint clustering maxK=300',
                   'BetaPoint Sampling; # Samples=300',
                   # target='condIncorrect',
                   # target='L2MPKI',
                   target='ipc',
                   )

    # truth = f'outputs/shotgun_continuous_point/{workload}.csv'

    # draw_betapoint(truth, pred)


if __name__ == '__main__':
    main()
