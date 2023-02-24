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
    wl_df['cpi'] = 1.0/wl_df['ipc']
    wl_df = wl_df.drop(['bmk', 'point', 'workload', 'ipc'], axis=1)

    # We also sort the vec_weight by point
    wl_js = dict(js[workload])
    # print(wl_js['points'])
    vec_weight = pd.DataFrame.from_dict(wl_js['points'], orient='index')
    vec_weight.index = vec_weight.index.astype(np.int64)
    vec_weight.columns = ['weight']
    vec_weight['weight'] = vec_weight['weight'].astype(np.float64)
    
    weight_metrics = np.matmul(vec_weight.values.reshape(1, -1), wl_df.values)
    return weight_metrics, wl_df.columns


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
    vec_weight = input_insts / np.sum(input_insts.values)
    weight_metric = np.matmul(vec_weight.values.reshape(1, -1), metrics.values)
    return weight_metric, metrics.columns


def compute_weighted_metrics(csv_path: str, js_path: str):
    df = pd.read_csv(csv_path, index_col=0)
    bmks = df['bmk'].unique()
    with open(js_path, 'r') as f:
        js = json.load(f)
    weighted = {}
    for bmk in bmks:
        df_bmk = df[df['bmk'] == bmk]
        workloads = df_bmk['workload'].unique()
        n_wl = len(workloads)
        if n_wl == 1:
            metrics, cols = proc_input(df_bmk, js, workloads[0])
        else:
            metrics, cols = proc_bmk(df_bmk, js, bmk)
        weighted[bmk] = metrics[0]
    weighted = pd.DataFrame.from_dict(weighted, orient='index', columns=cols)
    print(weighted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='specify results top directory and json')
    parser.add_argument('-r', '--results', action='store', required=True, help='results generated from batch.py')
    parser.add_argument('-j', '--json', action='store', required=True, help='json file containing weight info')
    args = parser.parse_args()
    compute_weighted_metrics(args.results, args.json)
