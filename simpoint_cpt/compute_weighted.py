import utils as u
import utils.common as c
import utils.target_stats as t
import json
import pandas as pd
import numpy as np
import sys
import os.path as osp
from scipy.stats import gmean
from statistics import geometric_mean
import argparse
import re


def proc_input(wl_df: pd.DataFrame, js: dict, workload: str):
    # we implement the weighted metrics computation with the following formula:
    # weight = vec_weight matmul matrix_perf
    # (N, 1) = (1, W) matmul (W, N)
    # To make sure the matrix_perf is in the same order as the vec_weight,
    # we sort the matrix_perf by point
    assert type(wl_df['point'][0]) == np.int64
    wl_df = wl_df.sort_values(by=['point'])
    if 'ipc' in wl_df.columns:
        wl_df['cpi'] = 1.0/wl_df['ipc']

    # We also sort the vec_weight by point
    wl_js = dict(js[workload])
    # print(wl_js['points'])
    vec_weight = pd.DataFrame.from_dict(wl_js['points'], orient='index')

    # convert string index into int64
    vec_weight.index = vec_weight.index.astype(np.int64)
    # select only existing points
    vec_weight = vec_weight.loc[wl_df['point']]
    # make their sum equals 1.0
    vec_weight.columns = ['weight']

    vec_weight['weight'] = vec_weight['weight'].astype(np.float64)
    coverage = np.sum(vec_weight.values)
    vec_weight = vec_weight / coverage
    
    # Drop these auxiliary fields

    to_drop = {'bmk', 'point', 'workload', 'ipc'}
    to_drop = to_drop.intersection(set(wl_df.columns.to_list()))
    # print(set(wl_df.columns.to_list()))
    print(to_drop)
    wl_df = wl_df.drop(to_drop, axis=1)

    weight_metrics = np.matmul(vec_weight.values.reshape(1, -1), wl_df.values)
    weight_metrics_df = pd.DataFrame(weight_metrics, columns=wl_df.columns)
    # We have to process coverage here to avoid apply weight on top of weight
    weight_metrics_df['coverage'] = coverage
    return weight_metrics_df.values, weight_metrics_df.columns


def proc_bmk(bmk_df: pd.DataFrame, js: dict, bmk: str):
    # Similar to per-input proc, we view the instruction count as the weight
    # and compute weighted metrics with matrix multiplication
    workloads = bmk_df['workload'].unique()
    metric_list = [] 
    for wl in workloads:
        metrics, cols = proc_input(bmk_df[bmk_df['workload'] == wl], js, wl)
        metric_list.append(metrics)
        print(f'{bmk} {wl} {metrics} {cols}')
    metrics = np.concatenate(metric_list, axis=0)
    metrics = pd.DataFrame(metrics, columns=cols)

    input_dict = {}
    for workload in workloads:
        if workload.startswith(workload):
            input_dict[workload] = int(js[workload]['insts'])
    input_insts = pd.DataFrame.from_dict(input_dict, orient='index', columns=['insts'])
    # make their sum equals 1.0
    vec_weight = input_insts / np.sum(input_insts.values)
    weight_metric = np.matmul(vec_weight.values.reshape(1, -1), metrics.values)
    return weight_metric, metrics.columns

def glob_weighted_stats_from_csv(batch_csv,simpoints):
    batch_df = pd.read_csv(batch_csv, index_col=0)
    weight_dict = json.load(open(simpoints))
    stat_tree = {}
    for index,row in batch_df.iterrows() :
        bmk = row['bmk']
        workload = row['workload']
        point = row['point']
        stat = row['ipc']
        weight = float(weight_dict[workload]['points'][str(point)])
        if bmk not in stat_tree:
            stat_tree[bmk] = {}
        if workload not in stat_tree[bmk]:
            stat_tree[bmk][workload] = {}
        if stat is not None:
            # print(bmk, workload, point)
            # print(weight, stat)
            assert int(point) not in stat_tree[bmk][workload]
            stat_tree[bmk][workload][int(point)] = (weight, stat)
    for bmk in stat_tree:
        for workload in stat_tree[bmk]:
            if len(stat_tree[bmk][workload]):
                df = pd.DataFrame.from_dict(stat_tree[bmk][workload], orient='index')
                df.columns = ['weight', 'ipc']
                stat_tree[bmk][workload] = df
            else:
                stat_tree[bmk].pop(workload)
                if not len(stat_tree[bmk]):
                    stat_tree.pop(bmk)
    return stat_tree


def compute_weighted_cpi(batch_csv,simpoints,var='06',
                         clock_rate=2 * 10**9, min_coverage=0.0, blacklist=[], whitelist=[],
                         merge_benckmark=True, output_csv='default.csv',
                         divide_intfp=False):
    workload_dict = {}
    bmk_stat = {}
    tree = glob_weighted_stats_from_csv(batch_csv,simpoints)
    with open(simpoints) as jf:
        js = json.load(jf)
        #print(js.keys())
    times = {}
    for bmk in tree:
        if bmk in blacklist:
            continue
        if len(whitelist) and bmk not in whitelist:
            continue
        cpis = []
        weights = []
        time = 0
        coverage = 0
        count = 0
        for workload, df in tree[bmk].items():
            selected = dict(js[workload])
            keys = [int(x) for x in selected['points']]
            keys = [x for x in keys if x in df.index]
            df = df.loc[keys]
            cpi, weight = u.weighted_cpi(df)
            weights.append(weight)
            workload_dict[workload] = {}
            workload_dict[workload]['CPI'] = cpi
            workload_dict[workload]['IPC'] = 1.0/cpi
            workload_dict[workload]['Coverage'] = weight
            # merge multiple sub-items of a benchmark
            if merge_benckmark:
                insts = int(js[workload]['insts'])
                workload_dict[workload]['TotalInst'] = insts
                workload_dict[workload]['PredictedCycles'] = insts*cpi
                seconds = insts*cpi / clock_rate
                workload_dict[workload]['PredictedSeconds'] = seconds
                time += seconds
                coverage += weight
                count += 1
        if merge_benckmark:
            bmk_stat[bmk] = {}
            bmk_stat[bmk]['time'] = time
            ref_time = c.get_spec_ref_time(bmk, var)
            assert ref_time is not None
            bmk_stat[bmk]['ref_time'] = ref_time
            bmk_stat[bmk]['score'] = ref_time / time
            bmk_stat[bmk]['Coverage'] = coverage/count
    df = pd.DataFrame.from_dict(workload_dict, orient='index')
    workload_dict = df
    if merge_benckmark:
        df = pd.DataFrame.from_dict(bmk_stat, orient='index')
        bmk_stat = df
        excluded = df[df['Coverage'] <= min_coverage]
        df = df[df['Coverage'] > min_coverage]
        df.to_csv(osp.join('results', output_csv))
        if divide_intfp:
            intdf = df.loc[u.spec_bmks['06']['int']]
            print('================ SPECint ================')
            print(intdf)
            print('Estimated score @ 2GHz:', geometric_mean(intdf['score']))
            print('Estimated score per GHz:', geometric_mean(intdf['score'])/(clock_rate/(10**9)))
            fpdf = df.loc[['bwaves','gamess','milc','zeusmp','gromacs','cactusADM','leslie3d','namd','dealII','soplex','povray','calculix','GemsFDTD','tonto','lbm','wrf','sphinx3']]
            print('================ SPECfp =================')
            print(fpdf)
            print('Estimated score @ 2GHz:', geometric_mean(fpdf['score']))
            print('Estimated score per GHz:', geometric_mean(fpdf['score'])/(clock_rate/(10**9)))
        else:
            print(df)
            print('Estimated score @ 2GHz:', geometric_mean(df['score']))
            print('Estimated score per GHz:', geometric_mean(df['score'])/(clock_rate/(10**9)))
            print('Excluded because of low coverage:', list(excluded.index))

def compute_weighted_metrics(csv_path: str, js_path: str, out_csv: str, args):
    df = pd.read_csv(csv_path, index_col=0)
    bmks = df['bmk'].unique()
    with open(js_path, 'r') as f:
        js = json.load(f)
    weighted = {}
    for bmk in bmks:
        if bmk not in u.spec_bmks['06']['int'] and args.int_only:
            continue
        if bmk not in u.spec_bmks['06']['float'] and args.fp_only:
            continue
        df_bmk = df[df['bmk'] == bmk]
        workloads = df_bmk['workload'].unique()
        n_wl = len(workloads)
        if n_wl == 1:
            metrics, cols = proc_input(df_bmk, js, workloads[0])
        else:
            metrics, cols = proc_bmk(df_bmk, js, bmk)
        weighted[bmk] = metrics[0]
    weighted = pd.DataFrame.from_dict(weighted, orient='index', columns=cols)
    if 'cpi' in weighted.columns:
        weighted = weighted.sort_values(by='cpi', ascending=False)
    else:
        weighted = weighted.sort_index()
    print(weighted)
    if out_csv is not None:
        if out_csv.startswith('/') or out_csv.startswith('.'):
            weighted.to_csv(out_csv)
        else:
            weighted.to_csv(osp.join('results', 'weighted_cpi.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='specify results top directory and json')
    parser.add_argument('-r', '--results', action='store', required=True, help='results generated from batch.py')
    parser.add_argument('-j', '--json', action='store', required=True, help='json file containing weight info')
    parser.add_argument('-o', '--output', action='store', required=False, help='csv file to stall results')
    parser.add_argument('-I', '--int-only', action='store_true', required=False, help='only process int')
    parser.add_argument('-F', '--fp-only', action='store_true', required=False, help='only process fp')
    parser.add_argument('-s', '--score', action='store', required=False, help='csv file to stall weighted score results')
    args = parser.parse_args()
    compute_weighted_metrics(args.results, args.json, args.output, args)
    if args.score:
        compute_weighted_cpi(args.results,
                             args.json,
                             var='06',
                             output_csv=args.score)
