import pandas as pd
import numpy as np
import utils as u
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import os.path as osp
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

def draw():
    results = {
        # "gem5-larger": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-larger-weighted.csv",
        # "gem5-larger-sq84": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-larger-sq84-weighted.csv",

        # "GEM5-Default": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-normal-weighted.csv",
        # "GEM5-ROB400": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-huge-weighted.csv",

        # For int all and bzip2
        "GEM5-base": osp.expandvars("$n/gem5-results/base-topdown-raw-weighted.csv"),
        # "GEM5-base2": osp.expandvars("$n/gem5-results/base-topdown-raw-weighted.csv"),

        # For bzip2
        "GEM5-redo": osp.expandvars("$n/gem5-results/base-redo-raw-weighted.csv"),

        # "GEM5-wide-squash": osp.expandvars("$n/gem5-results/base-squash24-raw-weighted.csv"),

        # For bzip2
        "GEM5-squash-1cycle": osp.expandvars("$n/gem5-results/base-squash-2c-weighted.csv"),

        # with multi pref
        # "GEM5-normal": "/nfs-nvme/home/zhouyaoyang/gem5-results/normal-multi-pref-weighted.csv",
        # "GEM5-ROB400": "/nfs-nvme/home/zhouyaoyang/projects/gem5_data_proc/topdown/temp_results/gem5-huge-multipref-weighted.csv",

        # top2 weighted:
        # "xs": "/nfs-nvme/home/zhouyaoyang/gem5-results/xs-topdown-l1.csv",
        # "gem5": "/nfs-nvme/home/zhouyaoyang/gem5-results/gem5-topdown-l1.csv",
        # "xs": "temp_results/xs-topdown-cpts.csv",
        # "gem5": "temp_results/gem5-topdown-cpts.csv",

        # GemsFDTD
        # "GemsFDTD修复前": "/nfs-nvme/home/zhouyaoyang/gem5-results/GemsFDTD-buggy.csv",
        # "GemsFDTD修复后": "/nfs-nvme/home/zhouyaoyang/gem5-results/GemsFDTD-fixed.csv",
    }

    configs = list(results.keys())

    # colors = ['#83639F', '#EA7827', '#C22F2F', '#449945', '#1F70A9']
    # edge_colors = ['#63437F', '#CA5807', '#A20F0F', '#247925', '#005099']
    color_types = 20
    if color_types == 20:
        cmap = plt.get_cmap('tab20')
        color_index = np.arange(0, 1, 1.0/20)
        colors = [cmap(c) for c in color_index]
        hatches = [None] * 20
    else:
        cmap = plt.get_cmap('tab10')
        color_index = np.arange(0, 1, 1.0/10)
        colors = [cmap(c) for c in color_index] * 3
        hatches = [None] * 10 + ['//']*10 + ['|']*10

    # print(colors)

    n_conf = len(configs)
    # Draw stacked bar chart for each simulator
    width = 0.8/n_conf
    # set figure size:

    eng_font = 'TimesNewRoman2'
    chn_font = 'SimSun'

    # mpl.use('pgf') # stwich backend to pgf
    # plt.rcParams.update({
    # "text.usetex": True,# use default xelatex
    # "pgf.rcfonts": False,# turn off default matplotlib fonts properties
    # "pgf.preamble": [
    #      r'\usepackage{fontspec}',
    #      r'\setmainfont{TimesNewRoman2}',# EN fonts Romans
    #      r'\usepackage{xeCJK}',# import xeCJK
    #      r'\setCJKmainfont{TimesNewRoman2}',# set CJK fonts as SimSun
    #      r'\xeCJKsetup{CJKecglue=}',# turn off one space between CJK and EN fonts
    #      ]
    # })

    mpl.rcParams['font.family'] = [eng_font]

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
    # fig.set_size_inches(5.77, 3.56)
    # fig.set_size_inches(5.77, 3.0)
    fig.set_size_inches(7.0, 3.0)
    # fig.set_size_inches(7.0, 5.0)

    x = None
    have_set_label = False

    rename_map= {
        'NoStall': None,

        # Core
        'LongExecute': None,
        'InstNotReady': None,

        # Memory
        'LoadL1Bound': None,
        'LoadL2Bound': None,
        'LoadL3Bound': None,
        'LoadMemBound': None,
        'StoreL1Bound': None,
        'StoreL2Bound': None,
        'StoreL3Bound': None,
        'StoreMemBound': None,
        'DTlbStall': None,

        # Frontend
        'IcacheStall': 'ICacheBubble',
        'ITlbStall': 'ITlbBubble',
        'FragStall': 'FragmentBubble',

        # BP
        'BpStall': 'MergeBadSpecBubble',
        'SquashStall': 'MergeBadSpecBubble',
        'InstMisPred': 'MergeBadSpecBubble',
        'InstSquashed': 'BadSpecInst',

        # BP + backend
        'CommitSquash': 'BadSpecWalking',

        # Unclassified:
        'SerializeStall': None,
        'TrapStall': None,
        'IntStall': 'MergeMisc',
        'ResumeUnblock': 'MergeMisc',
        'FetchBufferInvalid': 'MergeMisc',
        'OtherStall': 'MergeMisc',
        'OtherFetchStall': 'MergeMisc',
    }

    highlight_hatches = {
        'BadSpecWalking': '///',
        'BadSpecBubble': '|||',
        # 'FragmentBubble': '|||',
        # 'LoadMemBound': '\\\\\\',
    }

    dfs = [pd.read_csv(results[sim_conf], index_col=0) for sim_conf in results]
    common_bmk = list(set.intersection(*[set(df.index) for df in dfs]))
    # common_bmk = u.spec_bmks['06']['int']
    common_bmk = ['bzip2', 'sjeng']
    print(common_bmk)
    dfs = [df.loc[common_bmk] for df in dfs]
    sorted_cols = []
    for sim_conf, df in zip(results, dfs):
        # Merge df columns according to the rename map if value starting with 'Merge'
        to_drop = []
        for k in rename_map:
            if rename_map[k] is not None:
                if rename_map[k].startswith('Merge'):
                    merged = rename_map[k][5:]
                    if merged not in df.columns:
                        df[merged] = df[k]
                        sorted_cols.append(merged)
                    else:
                        df[merged] += df[k]
                else:
                    df[rename_map[k]] = df[k]
                    sorted_cols.append(rename_map[k])

                to_drop.append(k)
            else:
                sorted_cols.append(k)

        df = df.drop(columns=to_drop)

        # df = pd.read_csv(results[sim_conf], index_col=0)

        # if 'layer2_memory_bound-non_tlb' in df.columns:
        #     df['layer2_memory_bound'] = df['layer2_memory_bound-non_tlb'] + df['layer2_memory_bound-tlb']
        #     df.drop(columns=['layer2_memory_bound-non_tlb', 'layer2_memory_bound-tlb'], inplace=True)

        # drop non-numerical columns
        # df = df.drop(columns=[col for col in df.columns if not col.startswith(
        #     'layer') and not col.startswith('ipc') and not col.startswith('cpi')])

        # if 'ipc' in df.columns:
        #     df_cpi = 1.0/df['ipc']
        #     df = df.mul(df_cpi, axis=0)
        # elif 'cpi' in df.columns:
        #     df_cpi = df['cpi']
            # df = df.mul(df_cpi, axis=0)
            # df = df.div(df_cpi, axis=0)
        df = df.div(df['Insts'], axis=0)
        
        # df = df.drop(columns=[col for col in df.columns if not col.startswith('layer')])
        df = df.drop(columns=['Cycles', 'Insts', 'cpi', 'coverage'])

        # scale each row to make them sum to 1
        # df = df.div(df.sum(axis=1), axis=0)

        # add rows into one raw
        # df.loc['int overall'] = df.sum(axis=0)
        # df = df.loc[['int overall']]

        # scale each row to make them sum to 1
        # df = df.div(df.sum(axis=1), axis=0)

        print('CPI stack sum', df.sum(axis=1))
        # print(df)

        # draw stacked bar chart
        bottom = np.zeros(len(df))
        highest = 0.0
        if x is None:
            x = np.arange(len(df), dtype=float)
        for component, color, default_hatch in zip(sorted_cols, colors[:len(df.columns)], hatches[:len(df.columns)]):
            if component in highlight_hatches:
                hatch = highlight_hatches[component]
            else:
                hatch = default_hatch 
            if have_set_label:
                label = None
            else:
                label = component
            p = ax.bar(x, df[component], bottom=bottom,
                       width=width, color=color, label=label, edgecolor='black', hatch=hatch)
            highest = max(highest, max(bottom + df[component]))
            bottom += df[component]
        x += width
        have_set_label = True
    # replace x tick labels with df.index with rotation
    ax.set_xticks(x - width * len(results) / n_conf - 0.25)
    ax.set_xticklabels(df.index, rotation=0)
    ax.set_yticklabels([''])
    ax.tick_params(left=False, bottom=False)

    # # set the transparency of frame of legend
    handles, labels = plt.gca().get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), fancybox=True,
              framealpha=0.3,
              loc='lower right',
              ncol=2,
            #   loc='best',
            #   ncol=3,
              )
    # ax.set_title('GEM5 <-- Left, Right --> XS master')
    if n_conf == 2:
        ax.set_title(f'{configs[0]} <-- 左, 右 --> {configs[1]}')
    # ax.set_ylim(0, 4.0)
    # ax.set_ylim(0, highest * 2.2)
    ax.set_ylim(0, highest * 1.2)
    ax.set_xlim(-2.5, max(5.5, len(df)) * 2)

    tag = 'bzip2-sjeng-covered-stack'
    # tag = 'int-raw-topdown'
    fig.savefig(osp.join('results', f'{tag}.png'), bbox_inches='tight', pad_inches=0.05, dpi=200)
    fig.savefig(osp.join('results', f'{tag}.svg'), bbox_inches='tight', pad_inches=0.05)
    fig.savefig(osp.join('results', f'{tag}.pdf'), bbox_inches='tight', pad_inches=0.05)

    # drawing = svg2rlg(osp.join('results', f'{tag}.svg'))
    # renderPDF.drawToFile(drawing, osp.join('results', f'{tag}.pdf'))
    

if __name__ == '__main__':
    draw()
