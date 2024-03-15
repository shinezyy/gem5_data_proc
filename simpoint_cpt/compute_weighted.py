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
clock_rate = 3 * 10**9
reftime_js = {}

def proc_input(wl_df: pd.DataFrame, js: dict, workload: str):
    # we implement the weighted metrics computation with the following formula:
    # weight = vec_weight matmul matrix_perf
    # (N, 1) = (1, W) matmul (W, N)
    # To make sure the matrix_perf is in the same order as the vec_weight,
    # we sort the matrix_perf by point
    assert type(wl_df['point'].values[0]) == np.int64
    wl_df = wl_df.sort_values(by=['point'])
    # We also sort the vec_weight by point
    print('Processing bmk input', workload)
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
    try:
        vec_weight = vec_weight.loc[wl_df['point']]
    except KeyError:
        print(wl_df)
        if 0 in wl_df['point'].values:
            print(f"Ignore checkpoint 0 for {workload}")
            wl_df = wl_df[wl_df['point'] != 0]
            vec_weight = vec_weight.loc[wl_df['point']]
        else:
            print(f'KeyError: {workload}')
            print(vec_weight)
            print(wl_df['point'])
            raise KeyError
    print(vec_weight.shape)
    # make their sum equals 1.0
    vec_weight.columns = ['weight']

    vec_weight['weight'] = vec_weight['weight'].astype(np.float64)
    coverage = np.sum(vec_weight.values)
    vec_weight = vec_weight / coverage
    
    # Drop these auxiliary fields

    wl_df['weight'] = vec_weight.values
    wl_df.to_csv(osp.join('results', f'{workload}_raw.csv'))
    to_drop = {'bmk', 'point', 'workload', 'ipc', 'weight'}
    to_drop = to_drop.intersection(set(wl_df.columns.to_list()))
    # print(set(wl_df.columns.to_list()))
    # print(to_drop)
    wl_df = wl_df.drop(to_drop, axis=1)

    weight_metrics = np.matmul(vec_weight.values.reshape(1, -1), wl_df.values)
    decomposed = pd.DataFrame(wl_df.values * vec_weight.values, columns=wl_df.columns, index=wl_df.index)
    print(decomposed)  # decomposed
    decomposed['weight'] = vec_weight.values
    decomposed.to_csv(osp.join('results', f'{workload}_decomposed.csv'))
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
    print('Processing bmk', bmk)
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
    dirty_bmk_pattern = re.compile(r'(?P<name>\w+)-(?P<dirty>\d)')
    for bmk in bmks:
        m = dirty_bmk_pattern.match(bmk)
        pure_name = bmk
        if m:
            pure_name = m.group('name')
        if pure_name not in u.spec_bmks['06']['int'] and args.int_only:
            print(f'{bmk} not in int list')
            continue
        if pure_name not in u.spec_bmks['06']['float'] and args.fp_only:
            print(f'{bmk} not in fp list')
            continue
        df_bmk = df[df['bmk'] == bmk]
        workloads = df_bmk['workload'].unique()
        n_wl = len(workloads)
        print(workloads)
        if n_wl == 1:
            metrics, cols = proc_input(df_bmk, js, workloads[0])
        else:
            metrics, cols = proc_bmk(df_bmk, js, bmk)
        weighted[bmk] = metrics[0]
    weighted_df = pd.DataFrame.from_dict(weighted, orient='index', columns=cols)

    bmks_cleaned = []
    for bmk in weighted_df.index:
        m = dirty_bmk_pattern.match(bmk)
        if m:
            bmks_cleaned.append(m.group('name'))
        else:
            bmks_cleaned.append(bmk)
    weighted_df.index = bmks_cleaned
    pd.set_option("display.precision", 3)
    print(bmks_cleaned)

    if 'cpi' in weighted_df.columns:
        weighted_df = weighted_df.sort_values(by='cpi', ascending=False)
    else:
        weighted_df = weighted_df.sort_index()
    print(weighted_df)
    if out_csv is not None:
        weighted_df.to_csv(out_csv)
    if args.score:
        score = {}
        for bmk in weighted_df.index:
            if not score.get(bmk):
                score[bmk] = {}
            print(weighted_df.loc[bmk])
            score[bmk]['time'] = float(weighted_df.loc[bmk, 'time'])
            score[bmk]['ref_time'] = float(reftime_js[bmk])
            score[bmk]['score'] = score[bmk]['ref_time'] / score[bmk]['time']
            score[bmk]['coverage'] = weighted_df.loc[bmk, 'coverage']
        score['mean'] = {
            'time':0,
            'ref_time':0,
            'score': geometric_mean([x[1]['score'] for x in score.items()]),
            'coverage':0
        }
        score_col = ['time','ref_time','score','coverage']
        score = pd.DataFrame.from_dict(score, orient='index', columns=score_col)
        score = score.sort_values(by='score', ascending=False)
        score['score'] = score['score']/(clock_rate/(10**9))
        print(score)
        try:
            if args.int_only:
                intdf = score.loc[u.spec_bmks[spec_v]['int']]
            elif args.fp_only:
                fpdf = score.loc[u.spec_bmks[spec_v]['float']]
            else:# defalut
                intdf = score.loc[u.spec_bmks[spec_v]['int']]
                fpdf = score.loc[u.spec_bmks[spec_v]['float']]
            print(args.score)
            print(f'================ SPEC{spec_v} =================')
            # print(score)
            if not args.fp_only:
                print(f'================ Int =================')
                print(intdf)
                print('Estimated Int score per GHz:', geometric_mean(intdf['score']))
                print(f'Estimated Int score @ {clock_rate/(10**9)}GHz:', geometric_mean(intdf['score'])*(clock_rate/(10**9)))
            if not args.int_only:
                print(f'================ FP =================')
                print(fpdf)
                print('Estimated FP score per GHz:', geometric_mean(fpdf['score']))
                print(f'Estimated FP score @ {clock_rate/(10**9)}GHz:', geometric_mean(fpdf['score'])*(clock_rate/(10**9)))
            if not args.int_only and not args.fp_only:
                print(f'================ Overall =================')
                print('Estimated overall score per GHz:', score.loc['mean','score'])
                print(f'Estimated overall score @ {clock_rate/(10**9)}GHz:', score.loc['mean','score']*(clock_rate/(10**9)))
            if args.score is not None:
                score.to_csv(args.score)
        except:
            warnings.warn('spec result incomplete, scoring stop, print partial items')
            for bmk in u.spec_bmks[spec_v]['int'] + u.spec_bmks[spec_v]['float']:
                print(bmk)
                if bmk not in score.index:
                    print(f'Int {bmk} missing')
            int_list = [x for x in u.spec_bmks[spec_v]['int'] if x in score.index]
            fp_list = [x for x in u.spec_bmks[spec_v]['float'] if x in score.index]
            intdf = score.loc[int_list]
            fpdf = score.loc[fp_list]
            print(f'================ Int =================')
            print(intdf)
            print(f'================ FP =================')
            print(fpdf)
            if args.score is not None:
                score.to_csv(args.score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='specify results top directory and json')
    parser.add_argument('-r', '--results', action='store', required=True, help='results generated from batch.py')
    parser.add_argument('-j', '--json', action='store', required=True, help='json file containing weight info')
    parser.add_argument('-o', '--output', action='store', required=False, help='csv file to stall results')
    parser.add_argument('-I', '--int-only', action='store_true', required=False, help='only process int')
    parser.add_argument('-F', '--fp-only', action='store_true', required=False, help='only process fp')
    parser.add_argument('-s', '--score', action='store', required=False, help='csv file to stall weighted score results')
    parser.add_argument('-c', '--clock', action='store', required=False, default=3, help='simulation clock rate(GHz)')
    parser.add_argument('-v', '--spec-version', action='store', required=False, default='06',
                        help='spec version, default is 06')
    args = parser.parse_args()
    clock_rate = float(args.clock) * 10**9
    if args.score:
        spec_v = args.spec_version
        path = os.path.abspath(os.path.dirname(sys.argv[0]))
        with open(path + f'/resources/spec{spec_v}_reftime.json', 'r') as f:
            reftime_js = json.load(f)
    
    compute_weighted_metrics(args.results, args.json, args.output, args)
