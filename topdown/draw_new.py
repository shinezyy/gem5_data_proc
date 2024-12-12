# Usage: python3 topdown/draw_new.py -f=1 -p  -t1=spec_ideal_numBr4 -t2=spec_ideal_ftb64_numBr6
# Usage: python3 topdown/draw_new.py -f=3 -c=Frontend -t1=spec_ideal_numBr4 -t2=spec_ideal_ftb64_numBr6

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils.spec_info import spec_bmks
from topdown_stat_map import *
import matplotlib.colors as mcolors


def parse_args():
    parser = argparse.ArgumentParser(description='Draw topdown analysis chart')
    parser.add_argument('-f', '--level', type=int, choices=[1, 2, 3], default=3, help='Level of detail (1: highest level, 3: most detailed)')
    parser.add_argument('-c', '--category', choices=['All', 'Frontend', 'Backend', 'BadSpec', 'NoStall'], default='All', help='Specific category to analyze')
    parser.add_argument('-t1', '--tag1', required=True, help='Tag for the first input file')
    parser.add_argument('-t2', '--tag2', help='Tag for the second input file (optional)')
    parser.add_argument('-p', '--percent', action='store_true', help='Output topdown percentage')
    parser.add_argument('-t', '--type', choices=['int', 'fp', 'all'], default='all', help='Output all benchmarks')
    return parser.parse_args()

def filter_columns(df, category, hierarchy):
    if category == 'All':
        return df
    
    columns_to_keep = []
    for key, value in hierarchy.items():
        if key == category:
            if isinstance(value, list):
                columns_to_keep.extend(value)
            elif isinstance(value, dict):
                for sub_value in value.values():
                    if isinstance(sub_value, list):
                        columns_to_keep.extend(sub_value)
                    elif isinstance(sub_value, dict):
                        for sub_sub_value in sub_value.values():
                            columns_to_keep.extend(sub_sub_value)
    return df[columns_to_keep]

def load_and_process_data(tag, all_benchmarks, category, level):
    """加载并处理单个数据文件"""
    df = pd.read_csv(f"results/{tag}-weighted.csv", index_col=0)
    rename_with_map(df, gem5_topdown_hierarchy, level)  # 按照level 过滤对应数据
    df = filter_columns(df, category, gem5_topdown_hierarchy)
    return df.reindex(all_benchmarks)

def preprocess_for_percentage(df):
    """预处理数据用于百分比计算
    1. 删除cpi列
    2. 添加base列
    """
    df = df.copy()
    if 'cpi' in df.columns:
        df = df.drop(columns=['cpi'])
    df.insert(0, 'base', 20000000)
    return df

def calculate_percentages(df):
    """计算topdown百分比
    使用 base + frontend + backend + badspec 作为总和基准
    """
    df = preprocess_for_percentage(df)
    
    # 获取所有列的总和作为基准
    total = df['base'] + df['Frontend'].fillna(0) + df['Backend'].fillna(0) + df['BadSpec'].fillna(0)
    
    # 计算每列的百分比
    percentages = pd.DataFrame()
    percentages['base_per'] = df['base'] / total
    
    # 计算其他列的百分比
    for col in df.columns:
        if col != 'base':
            percentages[f'{col}_per'] = df[col] / total
    
    return percentages * 100

def print_percentage_analysis(df1, df2=None, tag1=None, tag2=None):
    """打印百分比分析结果"""
    print(f"\n{tag1} 百分比:")
    df1_percent = calculate_percentages(df1)
    print(df1_percent.apply(lambda x: round(x, 2)))  # 只保留2位小数
    
    if df2 is not None and tag2 is not None:
        print(f"\n{tag2} 百分比:")
        df2_percent = calculate_percentages(df2)
        print(df2_percent.apply(lambda x: round(x, 2)))  # 只保留2位小数
        
        print(f"\n差异 ({tag2} - {tag1}):")
        diff = df2_percent - df1_percent
        print(diff.apply(lambda x: round(x, 2)))  # 只保留2位小数   

def plot_comparison(df1, df2, tag1, tag2, category, level, output_filename):
    """绘制对比图表"""
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    hatches = [None, '//', '|', '\\', '+', 'x', 'o', 'O', '.', '*'] * 3
    
    all_columns = list(df1.columns)
    if df2 is not None:
        all_columns.extend([col for col in df2.columns if col not in df1.columns])
    
    color_map = {col: colors[i % len(colors)] for i, col in enumerate(all_columns)}
    hatch_map = {col: hatches[i % len(hatches)] for i, col in enumerate(all_columns)}

    fig, ax = plt.subplots(figsize=(20, 10))
    bar_width = 0.35
    x = np.arange(len(df1.index))

    def plot_bars(df, offset):
        bottom = np.zeros(len(df))
        for column in df.columns:
            ax.bar(x + offset, df[column], bar_width, bottom=bottom,
                  color=color_map[column], hatch=hatch_map[column], edgecolor='black')
            bottom += df[column]

    plot_bars(df1, -bar_width/2 if df2 is not None else 0)
    if df2 is not None:
        plot_bars(df2, bar_width/2)

    title = f'{tag1}' if df2 is None else f'{tag1} <-- VS. --> {tag2}'
    ax.set_title(f'{title}\n{category} (Level {level})')
    ax.set_xlabel('SPECCPU 2006 Benchmarks')
    ax.set_ylabel('Slots')
    ax.set_xticks(x)
    ax.set_xticklabels(df1.index, rotation=45, ha='right')

    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[col],
                                   hatch=hatch_map[col],
                                   edgecolor='black', label=col)
                      for col in all_columns]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Png saved to {output_filename}")

def draw(args):
    # 确定基准测试集
    int_benchmarks = spec_bmks['06']['int']
    fp_benchmarks = spec_bmks['06']['float']
    all_benchmarks = {
        'int': int_benchmarks,
        'fp': fp_benchmarks,
        'all': int_benchmarks + fp_benchmarks
    }[args.type]
    
    # 加载数据
    df1 = load_and_process_data(args.tag1, all_benchmarks, args.category, args.level)
    df2 = None if args.tag2 is None else load_and_process_data(args.tag2, all_benchmarks, args.category, args.level)
    
    # 输出原始数据
    print(f"\n{args.tag1} 原始数据:")
    print(df1)
    if df2 is not None:
        print(f"\n{args.tag2} 原始数据:")
        print(df2)
    
    # 如果需要显示百分比，计算并显示
    if args.percent:
        print_percentage_analysis(df1, df2, args.tag1, args.tag2)
    
    # 绘制图表，存储在figure目录下
    output_filename = f"figure/{args.tag1}.png" if df2 is None else f"figure/{args.tag1}_vs_{args.tag2}.png"
    plot_comparison(df1, df2, args.tag1, args.tag2, args.category, args.level, output_filename)

if __name__ == '__main__':
    args = parse_args()
    draw(args)