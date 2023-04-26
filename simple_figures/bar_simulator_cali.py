# Draw a bar chart for laptop_results\xs-gem5-1205.csv

import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import os.path as osp
import numpy as np
import pandas as pd


def draw(input_csv, output_pdf):
    int_list = ['perlbench', 'bzip2', 'gcc', 'mcf', 'gobmk', 'hmmer',
                'sjeng', 'libquantum', 'h264ref', 'omnetpp', 'astar', 'xalancbmk']

    df = pd.read_csv(input_csv, encoding='utf-8', header=0, index_col=None)
    
    fp_list = [x for x in df['bmk'] if x not in int_list]
    int_list = sorted(int_list)
    fp_list = sorted(fp_list)

    df.set_index('bmk', inplace=True)
    int_slice = df.loc[int_list]
    fp_slice = df.loc[fp_list]

    empty_slice = pd.DataFrame(columns=df.columns)
    empty_slice.loc[0] = [None] * len(empty_slice.columns)

    concated = pd.concat([int_slice, empty_slice, fp_slice], axis=0)

    xtick_labels = int_list + [''] + fp_list

    # set font

    eng_font = 'Times New Roman'
    chn_font = 'SimSun'

    mpl.use('pgf') # stwich backend to pgf
    plt.rcParams.update({
    "text.usetex": True,# use default xelatex
    "pgf.rcfonts": False,# turn off default matplotlib fonts properties
    "pgf.preamble": [
         r'\usepackage{fontspec}',
         r'\setmainfont{Times New Roman}',# EN fonts Romans
         r'\usepackage{xeCJK}',# import xeCJK
         r'\setCJKmainfont{SimSun}',# set CJK fonts as SimSun
         r'\xeCJKsetup{CJKecglue=}',# turn off one space between CJK and EN fonts
         ]
    })

    # mpl.rcParams['font.family'] = [chn_font]

    mpl.rcParams['font.family'] = [eng_font]
    mpl.rcParams['font.sans-serif'] = [chn_font]


    # to compare: GEM5-score,Nanhu-score
    fig, ax = plt.subplots()
    fig.set_size_inches(5.77, 3.56)  # 0.618
    # fig.set_size_inches(5.77, 5.77*(1-0.618))  # 1-0.618

    # add shift for each bar
    width = 0.35
    x_tick_pos = np.arange(len(xtick_labels))
    # print(ticks)
    # print(concated)
    ax.bar(x_tick_pos - width/2 - 0.03, concated['GEM5-score'], width, label='GEM5 分数',
          edgecolor='black', color='grey', linewidth=0.4,
           )
    ax.bar(x_tick_pos + width/2 + 0.03, concated['Nanhu-score'], width, label='Nanhu 分数',
          edgecolor='tab:orange', color='none', linewidth=0.4,
           )

    ax.set_xticks(x_tick_pos)

    # ax.set_xlabel('Benchmark')
    ax.set_ylabel('SPECCPU估算分数')
    ax.set_title('GEM5 VS. 南湖架构')
    ax.set_xticklabels(xtick_labels, rotation=45, fontsize=10, rotation_mode='anchor', ha='right')
    ax.legend()

    plt.text(len(int_list)/2.0 - 3, -30, "SPECCPU.int", fontsize=12)

    plt.text(len(int_list) + 1 + len(fp_list)/2.0 - 3, -30, "SPECCPU.fp", fontsize=12)

    plt.tight_layout()

    fig.savefig(output_pdf, bbox_inches='tight')
    # fig.savefig(osp.join('figures', 'bar_simulator_cali.pdf'), bbox_inches='tight')


if __name__ == "__main__":
    draw(osp.join('laptop_results', 'xs-gem5-1205.csv'),
         osp.join('figures', 'gem5_cali_1205.pdf'))
    draw(osp.join('laptop_results', 'xs-gem5-0115.csv'),
         osp.join('figures', 'gem5_cali_0115.pdf'))
    draw(osp.join('laptop_results', 'xs-gem5-0206.csv'),
         osp.join('figures', 'gem5_cali_0206.pdf'))
    draw(osp.join('laptop_results', 'xs-gem5-0305.csv'),
         osp.join('figures', 'gem5_cali_0305.pdf'))
