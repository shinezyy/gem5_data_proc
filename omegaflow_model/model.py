import common as c
import target_stats as t
import os
import sys
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

data = {
    'F1': osp.join(c.env.get_stat_dir(), 'f1_base-full'),
    'OoO': osp.join(c.env.get_stat_dir(), 'ooo_4w-full'),
}


def get_tuples(arch: str, targets):
    path = data[arch]
    matrix = {}
    print(path)
    for d in os.listdir(path):
        stat_dir_path = osp.join(path, d)
        if not osp.isdir(stat_dir_path):
            continue
        f = c.find_stats_file(stat_dir_path)
        tup = c.get_stats(f, targets, re_targets=True)
        matrix[d] = tup
    df = pd.DataFrame.from_dict(matrix, orient='index')
    return df


def main():
    f1_matrix = get_tuples('F1', t.model_targets)
    ooo_matrix = get_tuples('OoO', t.ipc_target)
    ooo_matrix.columns = ['ideal_ipc']
    # print(ooo_matrix)
    matrix = pd.concat([f1_matrix, ooo_matrix], axis=1, sort=True)
    matrix['PPI'] = matrix['0.TotalPackets']/matrix['Insts']
    matrix.sort_values(['PPI'], inplace=True)
    print(matrix)

    fig, ax = plt.subplots()
    fig.set_size_inches(14, 5)

    colors = plt.get_cmap('Dark2').colors

    # vertical line
    for index, row in matrix.iterrows():
        ax.plot((row['PPI'], row['PPI']), (row['ipc'], row['ideal_ipc']),
                color='lightgrey',
                zorder=1)

    # OoO
    ooo = ax.scatter(matrix['PPI'], matrix['ideal_ipc'],   marker='v', color=colors[1],
                     zorder=3)

    # FF
    ff = ax.scatter(matrix['PPI'], matrix['ipc'],         marker='^', color=colors[0],
                    zorder=3,
                    )
    objs = [ooo, ff]
    legends = ['Idealized OoO', 'Forwardflow']

    start = 1.4
    end = 2.52
    dots = np.arange(start, end, 0.01)

    def draw_model(rate: float, color):
        obj, = ax.plot(dots, rate/dots, marker=',', color=color)
        objs.append(obj)
        legends.append(f'Transferring and consumption rate={rate}')
        for index, row in matrix.iterrows():
            if rate/row['PPI'] < row['ideal_ipc']:
                ax.scatter([row['PPI']], [rate/row['PPI']], marker='x', color=color, zorder=2)

    draw_model(6.0, color=colors[6])
    draw_model(5.0, color=colors[5])
    draw_model(4.0, color=colors[4])
    draw_model(3.5, color=colors[3])
    draw_model(3.1, color=colors[2])
    draw_model(2.5, color=colors[7])

    ax.set_xlim([start, end])

    ax.set_ylabel('IPC', fontsize=14)
    ax.set_xlabel('PPI: pointers per instruction', fontsize=14)
    ax.legend(objs, legends, fontsize=12)
    # plt.plot()
    plt.show()


if __name__ == '__main__':
    main()


