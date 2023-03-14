import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import os.path as osp

def draw():
    results = {
        # "gem5-larger": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-larger-weighted.csv",
        # "gem5-larger-sq84": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-larger-sq84-weighted.csv",
        # "gem5-normal": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-normal-weighted.csv",
        # "gem5-huge": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-huge-weighted.csv",
        # "gem5-huge-multipref": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-huge-multipref-weighted.csv",

        # top2 weighted:
        # "xs": "/nfs-nvme/home/zhouyaoyang/gem5-results/xs-topdown-l1.csv",
        # "gem5": "/nfs-nvme/home/zhouyaoyang/gem5-results/gem5-topdown-l1.csv",
        # "xs": "temp_results/xs-topdown-cpts.csv",
        # "gem5": "temp_results/gem5-topdown-cpts.csv",

        # GemsFDTD
        "GemsFDTD修复前": "/nfs-nvme/home/zhouyaoyang/gem5-results/GemsFDTD-buggy.csv",
        "GemsFDTD修复后": "/nfs-nvme/home/zhouyaoyang/gem5-results/GemsFDTD-fixed.csv",
    }

    configs = list(results.keys())

    # colors = ['#83639F', '#EA7827', '#C22F2F', '#449945', '#1F70A9']
    # edge_colors = ['#63437F', '#CA5807', '#A20F0F', '#247925', '#005099']
    cmap = plt.get_cmap('tab10')
    color_index = np.arange(0, 1, 1.0/10)
    colors = [cmap(c) for c in color_index]
    edge_colors = colors
    # print(colors)

    # Draw stacked bar chart for each simulator
    shift = 0.8
    width = 0.3
    # set figure size:

    eng_font = 'TimesNewRoman2'
    chn_font = 'SimSun'

    mpl.use('pgf') # stwich backend to pgf
    plt.rcParams.update({
    "text.usetex": True,# use default xelatex
    "pgf.rcfonts": False,# turn off default matplotlib fonts properties
    "pgf.preamble": [
         r'\usepackage{fontspec}',
         r'\setmainfont{TimesNewRoman2}',# EN fonts Romans
         r'\usepackage{xeCJK}',# import xeCJK
         r'\setCJKmainfont{SimSun}',# set CJK fonts as SimSun
         r'\xeCJKsetup{CJKecglue=}',# turn off one space between CJK and EN fonts
         ]
    })

    mpl.rcParams['font.family'] = [chn_font]

    # mpl.rcParams['font.family'] = [eng_font]
    # mpl.rcParams['font.sans-serif'] = [chn_font]

    # config = {
    #     'font.family': 'serif',
    #     'mathtext.fontset': 'stix',
    #     'font.serif': ['SimSun'],
    # }
    # mpl.rcParams.update(config)

    # mpl.rcParams['font.style'] = 'normal'

    fig, ax = plt.subplots()
    fig.set_size_inches(5.77, 3.56)

    x = None
    have_set_label = False
    for simulators in results:
        df = pd.read_csv(results[simulators], index_col=0)
        # drop non-numerical columns
        df = df.drop(columns=[col for col in df.columns if not col.startswith(
            'layer') and not col.startswith('ipc') and not col.startswith('cpi')])
        if 'ipc' in df.columns:
            df_cpi = 1.0/df['ipc']
            df = df.mul(df_cpi, axis=0)
        elif 'cpi' in df.columns:
            df_cpi = df['cpi']
            df = df.mul(df_cpi, axis=0)
        df = df.drop(columns=[col for col in df.columns if not col.startswith('layer')])
        # draw stacked bar chart
        bottom = np.zeros(len(df))
        if x is None:
            x = np.arange(len(df), dtype=float)
        for component, color, ec in zip(df.columns, colors[:len(df.columns)],
                                        edge_colors[:len(df.columns)]):
            if have_set_label:
                label = None
            else:
                label = component
            p = ax.bar(x, df[component], bottom=bottom, width=width, color=color, label=label, edgecolor=ec)
            bottom += df[component]
        x += width
        have_set_label = True
    # replace x tick labels with df.index with rotation
    # ax.set_xticks(x - width * len(results) / 2)
    # ax.set_xticklabels(df.index, rotation=90)

    # # set the transparency of frame of legend
    ax.legend(fancybox=True, framealpha=0.3, loc='upper left')
    # ax.set_title('GEM5 <-- Left, Right --> XS master')
    ax.set_title(f'{configs[0]} <-- 左, 右 --> {configs[1]}')

    fig.savefig(osp.join('results', 'GemsFDTD.pdf'), bbox_inches='tight', pad_inches=0.05)
    # plt.show()
    

if __name__ == '__main__':
    draw()