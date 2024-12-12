import argparse
import pandas as pd
import csv
from topdown_stat_map import *
from utils.spec_info import spec_bmks

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze topdown statistics')
    parser.add_argument('-f', '--level', type=int, choices=[1, 2, 3], default=3, help='Level of detail (1: highest level, 3: most detailed)')
    parser.add_argument('input_file', help='Input CSV file with topdown statistics')
    # parser.add_argument('output_file', help='Output CSV file for analysis results')
    return parser.parse_args()

def calculate_percentages(df, hierarchy, level):
    rename_map = create_rename_map(hierarchy, level)
    columns_to_use = [col for col in df.columns if col in rename_map]
    
    # 处理MergeBase和BadSpecInst
    icount = 20*10**6
    # print(f'Base count:\n{df["NoStall"]}')
    if 'BadSpecInst' not in df.columns:
        df['BadSpecInst'] = df['NoStall'] - icount
    else:
        df['BadSpecInst'] += df['NoStall'] - icount
    # print(f'Bad inst count:\n{df["BadSpecInst"]}')
    df['NoStall'] = icount
    
    data = df[columns_to_use + ['BadSpecInst']]
    
    if level < 3:
        merged_df = pd.DataFrame()
        for k, v in rename_map.items():
            if k in data.columns:
                if v not in merged_df.columns:
                    merged_df[v] = data[k]
                else:
                    merged_df[v] += data[k]
        
        # 将 BadSpecInst 合并到 MergeBadSpec 中
        if 'MergeBadSpec' in merged_df.columns:
            merged_df['MergeBadSpec'] += data['BadSpecInst']
        else:
            merged_df['MergeBadSpec'] = data['BadSpecInst']
        
        data = merged_df
    else:
        # 对于 level 3，我们需要确保 BadSpecInst 被包含在计算中
        if 'MergeBadSpec' in rename_map.values():
            bad_spec_key = next(k for k, v in rename_map.items() if v == 'MergeBadSpec')
            data[bad_spec_key] += data['BadSpecInst']
        else:
            # 如果没有 MergeBadSpec，我们可以直接添加 BadSpecInst 列
            data['BadSpecInst'] = data['BadSpecInst']
    
    total = data.sum(axis=1)
    return data.div(total, axis=0) * 100

def main():
    args = parse_args()
    df = pd.read_csv(args.input_file, index_col=0)
    
    hierarchy = gem5_topdown_hierarchy
    
    percentages = calculate_percentages(df, hierarchy, args.level)
    
    # 定义整数和浮点测试项
    int_benchmarks = spec_bmks['06']['int']
    fp_benchmarks = spec_bmks['06']['float']
    all_benchmarks = int_benchmarks + fp_benchmarks

    # 计算整数和浮点测试的平均值
    int_avg = percentages.loc[percentages.index.isin(int_benchmarks)].mean()
    fp_avg = percentages.loc[percentages.index.isin(fp_benchmarks)].mean()
    all_avg = percentages.loc[percentages.index.isin(all_benchmarks)].mean()

    # # 将平均值添加到 percentages DataFrame
    # percentages.loc['int_avg'] = int_avg
    # percentages.loc['fp_avg'] = fp_avg
    # percentages.loc['all_avg'] = all_avg
    
    # # 将结果写入 CSV 文件
    # percentages.to_csv(args.output_file)

    # print(f"Analysis results have been written to {args.output_file}")

    print(f"================ SPEC06 =================")
    print(f"================ Int =================")
    # printf columns base  Frontend Backend Spec
    print(f"{'benchmark':14}", end="")
    for column in percentages.columns:
        # column 删除Merge前缀
        column = column[5:]
        print(f"{column:9}", end="")
    print()
    for benchmark in int_benchmarks:
        if benchmark in percentages.index:
            print(f"{benchmark:12}", end="")
            for column in percentages.columns:
                print(f"{percentages.loc[benchmark, column]:9.3f}", end="")
            print()
    print(f"Estimated Int average:")
    print(f"{'IntAvg':12}", end="")
    for column in int_avg.index:
        print(f"{int_avg[column]:9.3f}", end="")

    print(f"\n================ FP =================")
    print(f"{'benchmark':14}", end="")
    for column in percentages.columns:
        # column 删除Merge前缀
        column = column[5:]
        print(f"{column:9}", end="")
    print()
    for benchmark in fp_benchmarks:
        if benchmark in percentages.index:
            print(f"{benchmark:12}", end="")
            for column in percentages.columns:
                print(f"{percentages.loc[benchmark, column]:9.3f}", end="")
            print()
    print(f"Estimated FP average:")
    print(f"{'FpAvg':12}", end="")
    for column in fp_avg.index:
        print(f"{fp_avg[column]:9.3f}", end="")

    print(f"\n================ Overall =================")
    print(f"Estimated overall average:")
    print(f"{'AllAvg':12}", end="")
    for column in all_avg.index:
        print(f"{all_avg[column]:9.3f}", end="")
    print()

if __name__ == "__main__":
    main()