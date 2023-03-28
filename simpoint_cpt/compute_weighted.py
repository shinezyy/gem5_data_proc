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
import os
import warnings

args = []
spec_v = '06'
clock_rate = 2 * 10**9
reftime_js = {}

def proc_input(wl_df: pd.DataFrame, js: dict, workload: str):
    # we implement the weighted metrics computation with the following formula:
    # weight = vec_weight matmul matrix_perf
    # (N, 1) = (1, W) matmul (W, N)
    # To make sure the matrix_perf is in the same order as the vec_weight,
    # we sort the matrix_perf by point
    assert type(wl_df['point'][0]) == np.int64
    wl_df = wl_df.sort_values(by=['point'])
    # We also sort the vec_weight by point
    wl_js = dict(js[workload])
    if 'ipc' in wl_df.columns:
        wl_df['cpi'] = 1.0/wl_df['ipc']
    if args.score:
        wl_df['time'] = int(wl_js['insts']) * wl_df['cpi'] / clock_rate
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
    # print(to_drop)
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
    time = 0
    for wl in workloads:
        metrics, cols = proc_input(bmk_df[bmk_df['workload'] == wl], js, wl)
        if args.score:
            time += metrics[0][np.where(cols.values == 'time')[0][0]]
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
    if args.score:
        weight_metric[0][np.where(cols.values == 'time')[0][0]] = time
    return weight_metric, metrics.columns


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
    weighted_df = pd.DataFrame.from_dict(weighted, orient='index', columns=cols)
    if 'cpi' in weighted_df.columns:
        weighted_df = weighted_df.sort_values(by='cpi', ascending=False)
    else:
        weighted_df = weighted_df.sort_index()
    print(weighted_df)
    if out_csv is not None:
        if out_csv.startswith('/') or out_csv.startswith('.'):
            weighted_df.to_csv(out_csv)
        else:
            weighted_df.to_csv(osp.join('results', 'weighted_cpi.csv'))
    if args.score:
        time_idx = np.where(cols.values == 'time')[0][0]
        coverage_idx = np.where(cols.values == 'coverage')[0][0]
        score = {}
        for bmk in weighted.items():
            if not score.get(bmk[0]) :
                score[bmk[0]] = {}
            score[bmk[0]]['time'] = float(bmk[1][time_idx])
            score[bmk[0]]['ref_time'] = float(reftime_js[bmk[0]])
            score[bmk[0]]['score'] = score[bmk[0]]['ref_time'] / score[bmk[0]]['time']
            score[bmk[0]]['coverage'] = bmk[1][coverage_idx]
        score['mean'] = {
            'time':0,
            'ref_time':0,
            'score': geometric_mean([x[1]['score'] for x in score.items()]),
            'coverage':0
        }
        score_col = ['time','ref_time','score','coverage']
        score = pd.DataFrame.from_dict(score, orient='index', columns=score_col)
        score = score.sort_values(by='score', ascending=False)
        try:
            if args.int_only:
                intdf = score.loc[u.spec_bmks[spec_v]['int']]
            elif args.fp_only:
                fpdf = score.loc[u.spec_bmks[spec_v]['float']]
            else:# defalut
                intdf = score.loc[u.spec_bmks[spec_v]['int']]
                fpdf = score.loc[u.spec_bmks[spec_v]['float']]
            print(f'================ SPEC{spec_v} =================')
            print(score)
            print(f'Estimated score @ {clock_rate}GHz:', score.loc['mean','score'])
            print('Estimated score per GHz:', score.loc['mean','score']/(clock_rate/(10**9)))
        except:
            warnings.warn('spec result incomplete, scoring stop')

        if args.score.startswith('/') or args.score.startswith('.'):
            score.to_csv(args.score)
        else:
            score.to_csv(osp.join('results', 'score.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='specify results top directory and json')
    parser.add_argument('-r', '--results', action='store', required=True, help='results generated from batch.py')
    parser.add_argument('-j', '--json', action='store', required=True, help='json file containing weight info')
    parser.add_argument('-o', '--output', action='store', required=False, help='csv file to stall results')
    parser.add_argument('-I', '--int-only', action='store_true', required=False, help='only process int')
    parser.add_argument('-F', '--fp-only', action='store_true', required=False, help='only process fp')
    parser.add_argument('-s', '--score', action='store', required=False, help='csv file to stall weighted score results')
    parser.add_argument('-c', '--clock', action='store', required=False, default=2, help='simulation clock rate(GHz)')
    args = parser.parse_args()
    clock_rate = float(args.clock) * 10**9
    if args.score:
        spec_v = args.json.split('/')[-1].split('_')[0][4:6]
        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        with open(path + f'/resources/spec{spec_v}_reftime.json', 'r') as f:
            reftime_js = json.load(f)
    
    compute_weighted_metrics(args.results, args.json, args.output, args)
