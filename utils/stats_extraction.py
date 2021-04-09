import os
import os.path as osp
import pandas as pd
import numpy as np
import utils.common as c
import utils.target_stats as t
import json
import re

def get_ipc(stat_path: str):
    targets = t.ipc_target
    stats = c.get_stats(stat_path, targets, insts=100*10**6, re_targets=True)
    return stats['ipc']

def stats_factory(targets, keys, prefix=''):
    def pattern_to_meta(p):
        return re.compile('.*\((\w.+)\).*').search(p).group(1)
    def keys_to_meta(ks):
        if type(keys) == list:
            if prefix == '':
                meta_keys = [pattern_to_meta(k) for k in ks]
            else:
                meta_keys = ks
        else:
            assert type(ks) == str
            if prefix == '':
                meta_keys = [pattern_to_meta(ks)]
            else:
                meta_keys = [ks]
        return meta_keys
    def get_stat(stat_path: str):
        # print(stat_path)
        if prefix == '':
            stats = c.get_stats(stat_path, targets, re_targets=True)
        else:
            assert prefix == 'xs_'
            stats = c.xs_get_stats(stat_path, targets, re_targets=True)
        if stats is not None and len(stats):
            res = {}
            meta_keys = keys_to_meta(keys)
            for k in meta_keys:
                res[k] = stats[k]
            return res
        else:
            return None
    return get_stat

def glob_stats_l2(path: str, fname = 'm5out/stats.txt'):
    stats_files = []
    for x in os.listdir(path):
        x_path = osp.join(path, x)
        if osp.isdir(x_path):
            for codename, f in glob_stats(x_path, fname):
                stats_files.append((f'{x}/{codename}', f))
    return stats_files

def glob_stats(path: str, fname = 'm5out/stats.txt'):
    stat_files = []
    for x in os.listdir(path):
        stat_path = osp.join(path, x, fname)
        if osp.isfile(stat_path) and osp.getsize(stat_path) > 10 * 1024:  # > 10K
            stat_files.append((x, stat_path))
    return stat_files

def glob_weighted_stats(path: str, get_stats_func, filtered=True,
        simpoints='/home51/zyy/expri_results/simpoints17.json',
        stat_file='m5out/stats.txt'):
    stat_tree = {}
    with open(simpoints) as jf:
        points = json.load(jf)
    print(points.keys())
    for workload in os.listdir(path):
        print('Extracting', workload)
        weight_pattern = re.compile(f'.*/{workload}_\d+_(\d+\.\d+([eE][-+]?\d+)?)/.*\.gz')
        workload_dir = osp.join(path, workload)
        bmk = workload.split('_')[0]
        if bmk not in stat_tree:
            stat_tree[bmk] = {}
        stat_tree[bmk][workload] = {}
        for point in os.listdir(workload_dir):
            if point not in points[workload]:
                continue
            point_dir = osp.join(workload_dir, point)
            stats_file = osp.join(point_dir, stat_file)
            weight = float(points[workload][str(point)])
            stats = get_stats_func(stats_file)
            # stat = get_func(stats_file)
            if stats is not None:
                stat_tree[bmk][workload][int(point)] = [weight] + [s for s in stats.values()]
        if len(stat_tree[bmk][workload]):
            df = pd.DataFrame.from_dict(stat_tree[bmk][workload], orient='index')
            df.columns = ['weight'] + [k for k in stats.keys()]
            stat_tree[bmk][workload] = df
            print(df)
        else:
            stat_tree[bmk].pop(workload)
            if not len(stat_tree[bmk]):
                stat_tree.pop(bmk)
    print(stat_tree)
    return stat_tree


def glob_weighted_cpts(path: str):
    stat_tree = {}
    for dir_name in os.listdir(path):
        weight_pattern = re.compile(f'(\w+)_(\d+)_(\d+\.\d+([eE][-+]?\d+)?)')
        workload_dir = osp.join(path, dir_name)
        print(dir_name)
        m = weight_pattern.match(dir_name)
        if m is not None:
            workload = m.group(1)
            point = m.group(2)
            weight = float(m.group(3))
            bmk = workload.split('_')[0]
            if bmk not in stat_tree:
                stat_tree[bmk] = {}
            if workload not in stat_tree[bmk]:
                stat_tree[bmk][workload] = {}
            stat_tree[bmk][workload][int(point)] = weight
    for bmk in stat_tree:
        for workload in stat_tree[bmk]:
            item = stat_tree[bmk][workload]
            df = pd.DataFrame.from_dict(item, orient='index')
            df.columns = ['weight']
            stat_tree[bmk][workload] = df
    return stat_tree



def coveraged(coverage: float, df: pd.DataFrame):
    df = df.sort_values('weight', ascending=False)
    i = 0
    cummulated = 0.0
    for row in df.iterrows():
        cummulated += row[1]['weight']
        i += 1
        if cummulated > coverage:
            break
    df = df.iloc[:i]
    return(cummulated, df)


def weighted_cpi(df: pd.DataFrame):
    if 'cpi' in df.columns:
        return np.dot(df['weight'], df['cpi']) / np.sum(df['weight']), np.sum(df['weight'])
    else:
        assert 'ipc' in df.columns
        return np.dot(df['weight'], 1.0/df['ipc']) / np.sum(df['weight']), np.sum(df['weight'])

def weighted_mpkis(df: pd.DataFrame):
    assert 'Insts' in df.columns
    assert 'branchMispredicts' in df.columns
    mpki = 1000 * np.true_divide(df['branchMispredicts'], df['Insts'])
    weighted = np.dot(df['weight'], mpki) / np.sum(df['weight'])
    return weighted

def xs_weighted_mpkis(df: pd.DataFrame,
    targets=[
        'BpWrong',
        'BpBWrong',
        'BpJWrong',
        'BpIWrong',
        'BpCWrong',
        'BpRWrong'
    ]):
    assert 'roq: commitInstr' in df.columns
    for t in targets:
        assert t in df.columns
    mpkis = [1000 * np.true_divide(df[t], df['roq: commitInstr']) for t in targets]
    weighted = [np.dot(df['weight'], m) / np.sum(df['weight']) for m in mpkis]
    return weighted


def gen_json(path, output):
    tree = glob_weighted_cpts(path)
    d = {}
    count = 0
    expected_coverage = 0.5
    for bmk in tree:
        for workload in tree[bmk]:
            d[workload] = {}
            coverage, selected = coveraged(expected_coverage, tree[bmk][workload])
            weights = selected['weight']
            for row in weights.index:
                count += 1
                d[workload][int(row)] = weights[row]
            print(d[workload])
    print(f'{count} points in total')
    with open(f'{output}_cover{expected_coverage}.json', 'w') as f:
        json.dump(d, f, indent=4)



if __name__ == '__main__':
    tree = glob_weighted_stats(
            '/home51/zyy/expri_results/omegaflow_spec17/of_g1_perf/',
            {'cpi': stats_factory(t.ipc_target, 'cpi')}
            )
    # gen_json('/home51/zyy/expri_results/nemu_take_simpoint_cpt_06',
    #         '/home51/zyy/expri_results/simpoints06')
