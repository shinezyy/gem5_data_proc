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


def main():
    f1_matrix = c.get_df(data, 'F1', t.model_targets, 'imagick_0')
    ooo_matrix = c.get_df(data, 'OoO', t.ipc_target, 'imagick_0')
    ooo_matrix.columns = ['ideal_ipc']
    # print(ooo_matrix)
    matrix = pd.concat([f1_matrix, ooo_matrix], axis=1, sort=True)
    matrix['PPI'] = matrix['0.TotalPackets']/matrix['Insts']
    matrix['PPI'] = matrix['PPI'] - 0.5
    matrix = matrix[['ipc', 'ideal_ipc', 'PPI']]
    matrix = matrix.append({
        'ipc': matrix['ipc'][0],
        'ideal_ipc': matrix['ideal_ipc'][0],
        'PPI': matrix['PPI'][0] - 0.6,
    },
        ignore_index=True,
    )
    matrix['ipc'] = matrix['ipc'] + 0.2
    matrix.sort_values(['PPI'], inplace=True)
    print(matrix)

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 5)


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

    start = 0.5
    end = 2.52
    dots = np.arange(start, end, 0.01)
    ax.set_xlim([start, end])
    ax.set_ylim([0.01, 5.0])

    def draw_model(rate: float, color):
        obj, = ax.plot(dots, rate/dots, marker=',', color=color)
        objs.append(obj)
        legends.append(f'Transferring and consumption rate={rate}')
        for index, row in matrix.iterrows():
            if rate/row['PPI'] < row['ideal_ipc']:
                ax.scatter([row['PPI']], [rate/row['PPI']], marker='x', color=color, zorder=2)

    high_rate = 4.5
    draw_model(high_rate, color=colors[6])
    # draw_model(5.0, color=colors[5])
    # draw_model(4.0, color=colors[4])
    # draw_model(3.5, color=colors[3])
    draw_model(3.1, color=colors[2])
    # draw_model(2.5, color=colors[7])

    # from original PPI to new
    src = int(0)
    dest = int(1)
    src_x = matrix['PPI'][src]
    src_y = 0.9*matrix['ipc'][src] + 0.1*matrix['ideal_ipc'][src]
    dest_x = matrix['PPI'][dest]
    dest_y = src_y
    mid_x = 0.5*src_x + 0.5*dest_x
    mid_y = dest_y
    margin = 0.05
    line_width = 0.01
    ax.arrow(src_x - margin, src_y,
             dest_x - src_x + 2*margin,  0,
             head_width=10*line_width,
             head_length=10*line_width,
             length_includes_head=True,
             color=colors[3],
             )
    plt.text(mid_x, mid_y + 0.1,
             'Move programs left\nby reducing $PPI$',
             horizontalalignment='center')

    # Increase Rate:
    src_x = 1.5
    src_y = 3.1/src_x
    dest_y = high_rate/src_x
    mid_x = src_x
    mid_y = 0.5*src_y + 0.5*dest_y
    margin = 0.05
    line_width = 0.01
    ax.arrow(src_x, src_y,
             0, dest_y - src_y,
             head_width=3 * line_width,
             head_length=30 * line_width,
             length_includes_head=True,
             color=colors[4],
             )
    plt.text(mid_x+0.02, mid_y-0.15,
             'Push curve up\nby increasing $Rate$',
             horizontalalignment='left',
             verticalalignment='top',
             )

    ax.set_ylabel('IPC', fontsize=14)
    ax.set_xlabel('PPI: pointers per instruction', fontsize=14)
    ax.legend(objs, legends, fontsize=12)
    # plt.plot()
    plt.show()


if __name__ == '__main__':
    main()


