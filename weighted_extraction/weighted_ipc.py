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

simpoints17 = '/home51/zyy/expri_results/simpoints.json'

def gen_coverage():
    tree = u.glob_weighted_stats(
            '/home51/zyy/expri_results/omegaflow_spec17/of_g1_perf/',
            u.single_stat_factory(t.ipc_target, 'cpi')
            )
    d = {}
    for bmk in tree:
        for workload in tree[bmk]:
            d[workload] = {}
            coverage, selected = u.coveraged(0.8, tree[bmk][workload])
            print(coverage)
            weights = selected['weight']
            for row in weights.index:
                d[workload][int(row)] = weights[row]
            print(d[workload])
    with open(simpoints, 'w') as f:
        json.dump(d, f, indent=4)


def get_insts(fname: str):
    assert osp.isfile(fname)
    p = re.compile('total guest instructions = (\d+)')
    with open(fname) as f:
        for line in f:
            m = p.search(line)
            if m is not None:
                return m.group(1)
    return None


def compute_weighted_cpi(ver, confs, base, simpoints, prefix, insts_file_fmt, stat_file,
        clock_rate, min_coverage=0.0, blacklist=[], whitelist=[], merge_benckmark=False):
    target = eval(f't.{prefix}ipc_target')
    workload_dict = {}
    bmk_stat = {}
    for conf, file_path in confs.items():
        workload_dict[conf] = {}
        bmk_stat[conf] = {}
        tree = u.glob_weighted_stats(
                file_path,
                u.single_stat_factory(target, 'ipc', prefix),
                simpoints=simpoints,
                stat_file=stat_file,
                )
        with open(simpoints) as jf:
            js = json.load(jf)
            print(js.keys())
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
                keys = [int(x) for x in selected]
                keys = [x for x in keys if x in df.index]
                df = df.loc[keys]
                cpi, weight = u.weighted_cpi(df)
                weights.append(weight)

                workload_dict[conf][workload] = {}
                workload_dict[conf][workload]['CPI'] = cpi
                workload_dict[conf][workload]['IPC'] = 1.0/cpi
                workload_dict[conf][workload]['Coverage'] = weight

                # merge multiple sub-items of a benchmark
                if merge_benckmark:
                    insts_file = insts_file_fmt.format(ver, workload)
                    insts = int(get_insts(insts_file))
                    workload_dict[conf][workload]['TotalInst'] = insts
                    workload_dict[conf][workload]['PredictedCycles'] = insts*cpi
                    seconds = insts*cpi / clock_rate
                    workload_dict[conf][workload]['PredictedSeconds'] = seconds
                    time += seconds
                    coverage += weight
                    count += 1

            if merge_benckmark:
                bmk_stat[conf][bmk] = {}
                bmk_stat[conf][bmk]['time'] = time
                ref_time = c.get_spec_ref_time(bmk, ver)
                assert ref_time is not None
                bmk_stat[conf][bmk]['ref_time'] = ref_time
                bmk_stat[conf][bmk]['score'] = ref_time / time
                bmk_stat[conf][bmk]['Coverage'] = coverage/count

    for conf in confs:
        print(conf, '='*60)
        df = pd.DataFrame.from_dict(workload_dict[conf], orient='index')
        workload_dict[conf] = df
        print(df)

        if merge_benckmark:
            df = pd.DataFrame.from_dict(bmk_stat[conf], orient='index')
            bmk_stat[conf] = df
            excluded = df[df['Coverage'] <= min_coverage]
            df = df[df['Coverage'] > min_coverage]
            print(df)
            print('Estimated score @ 1.5GHz:', geometric_mean(df['score']))
            print('Estimated score per GHz:', geometric_mean(df['score'])/(clock_rate/(10**9)))
            print('Excluded because of low coverage:', list(excluded.index))


    tests = []
    for conf in confs.keys():
        if conf == base:
            continue
        rel = workload_dict[conf]['IPC']/workload_dict[base]['IPC']
        print(f'{conf}  Mean relative performance: {gmean(rel)}')
        tests.append(rel)
    dfx = pd.concat(tests, axis=1)
    print('Relative performance:')
    print(dfx)


def gem5_spec2017():
    ver = '17'
    confs = {
            # 'O1': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH1S1G1Config',
            # 'O2': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH1S1G2CL0Config',
            # 'O1S0': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH1S0G1Config',
            # 'O1X': '/home51/zyy/expri_results/omegaflow_spec17/XOmegaH1S1G1Config',
            # 'O1*-': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH0S0G1Config',
            # 'F1-': '/home51/zyy/expri_results/omegaflow_spec17/FFS0Config',
            # 'F1H': '/home51/zyy/expri_results/omegaflow_spec17/FFH1Config',
            'F1': '/home51/zyy/expri_results/omegaflow_spec17/TypicalFFConfig',
            'F2': '/home51/zyy/expri_results/omegaflow_spec17/FFG2CL0CG1Config',
            'FullO3': '/home51/zyy/expri_results/omegaflow_spec17/FullWindowO3Config'
            }

    compute_weighted_cpi(
            ver=ver,
            confs=confs,
            base='FullO3',
            simpoints=f'/home51/zyy/expri_results/simpoints{ver}.json',
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
            clock_rate = 4 * 10**9,
            min_coverage = 0.75,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )


def xiangshan_spec2006():
    ver = '06'
    confs = {
            'XiangShan1': '/home/zyy/expri_results/xs_simpoint_batch/SPEC06_EmuTasksConfig',
            'XiangShan2': '/home/zyy/expri_results/xs_simpoint_batch/SPEC06_EmuTasksConfig',
            }

    compute_weighted_cpi(
            ver=ver,
            confs=confs,
            base='XiangShan1',
            simpoints=f'/home51/zyy/expri_results/simpoints{ver}.json',
            prefix = 'xs_',
            stat_file='simulator_err.txt',
            insts_file_fmt =
            '/bigdata/zyy/checkpoints_profiles/betapoint_profile_{}_fix_mem_addr/{}/nemu_out.txt',
            clock_rate = 1.5 * 10**9,
            min_coverage = 0.75,
            # blacklist = ['gamess'],
            merge_benckmark=True,
            )


if __name__ == '__main__':
    xiangshan_spec2006()
    # gem5_spec2017()
