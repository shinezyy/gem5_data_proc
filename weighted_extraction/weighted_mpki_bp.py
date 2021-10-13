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
import functools as ft

simpoints17 = '/home51/zyy/expri_results/simpoints.json'

def gen_coverage():
    tree = u.glob_weighted_stats(
            '/home51/zyy/expri_results/omegaflow_spec17/of_g1_perf/',
            u.stats_factory(t.ipc_target, 'cpi')
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


def compute_weighted_mpki(ver, confs, base, simpoints, prefix, insts_file_fmt, stat_file,
        clock_rate, min_coverage=0.0, blacklist=[], whitelist=[], merge_benckmark=False):
    target = eval(f't.{prefix}branch_targets')
    workload_dict = {}
    bmk_stat = {}
    for conf, file_path in confs.items():
        workload_dict[conf] = {}
        bmk_stat[conf] = {}
        if prefix == '':
            stats = ['cpus?\.(?:diewc|commit|iew)\.(branchMispredicts)', 'cpus?\.committed(Insts)', 'cpus?\.(?:diewc\.exec_|commit\.)(branches)', 'cpus?\.(ipc)']
        else:
            assert prefix == 'xs_'
            stats = target + ['(ipc)']
        tree = u.glob_weighted_stats(
                file_path,
                ft.partial(u.stats_factory, target, stats, prefix=prefix),
                whitelist,
                simpoints=simpoints,
                stat_file=stat_file,
                )
        with open(simpoints) as jf:
            js = json.load(jf)
            # print(js.keys())
        times = {}
        print(tree)
        for bmk in tree:
            if bmk in blacklist:
                continue
            if len(whitelist) and bmk not in whitelist:
                continue
            mpkis = []
            cpis = []
            weights = []
            time = 0
            coverage = 0
            count = 0
            total_misp = 0
            total_misp_b = 0
            total_misp_b_ubtb = 0
            total_misp_b_btb = 0
            total_misp_j = 0
            total_misp_i = 0
            total_misp_c = 0
            total_misp_r = 0
            total_rdc_sc = 0
            total_inst = 0
            total_branches = 0
            total_cycle = 0
            for workload, df in tree[bmk].items():
                # print(workload)
                selected = dict(js[workload])
                keys = [int(x) for x in selected]
                keys = [x for x in keys if x in df.index]
                df = df.loc[keys]
                # print(df)
                cpi, weight = u.weighted_cpi(df)
                if prefix == '':
                    (b_mpki, bpki, b_misrate) = u.weighted_mpkis(df)
                else:
                    assert prefix == 'xs_'
                    ([total_mpki, b_mpki, j_mpki, i_mpki, c_mpki, r_mpki], bpki, b_misrate, sc_rdc_mpki) = u.xs_weighted_mpkis(df)
                weights.append(weight)

                workload_dict[conf][workload] = {}
                workload_dict[conf][workload]['CPI'] = cpi
                workload_dict[conf][workload]['IPC'] = 1.0/cpi
                workload_dict[conf][workload]['MPKI_B'] = b_mpki
                workload_dict[conf][workload]['BPKI'] = bpki
                workload_dict[conf][workload]['MISRATE_B'] = b_misrate
                if prefix == 'xs_':
                    workload_dict[conf][workload]['MPKI'] = total_mpki
                    # workload_dict[conf][workload]['MPKI_B_UBTB'] = ubtb_b_mpki
                    # workload_dict[conf][workload]['MPKI_B_BTB'] = btb_b_mpki
                    workload_dict[conf][workload]['MPKI_J'] = j_mpki
                    workload_dict[conf][workload]['MPKI_I'] = i_mpki
                    workload_dict[conf][workload]['MPKI_C'] = c_mpki
                    workload_dict[conf][workload]['MPKI_R'] = r_mpki
                    workload_dict[conf][workload]['MPKI_RDC_SC'] = sc_rdc_mpki
                workload_dict[conf][workload]['Coverage'] = weight

                # merge multiple sub-items of a benchmark
                if merge_benckmark:
                    insts_file = insts_file_fmt.format(workload)
                    insts = int(get_insts(insts_file))
                    workload_dict[conf][workload]['TotalInst'] = insts
                    workload_dict[conf][workload]['PredictedCycles'] = insts*cpi
                    seconds = insts*cpi / clock_rate
                    workload_dict[conf][workload]['PredictedSeconds'] = seconds
                    time += seconds
                    total_inst += insts
                    total_cycle += insts * cpi
                    total_misp_b += insts*b_mpki/1000
                    total_branches += insts*bpki/1000
                    if prefix == 'xs_':
                        total_misp += insts*total_mpki/1000
                        # total_misp_b_ubtb += insts*ubtb_b_mpki/1000
                        # total_misp_b_btb += insts*btb_b_mpki/1000
                        total_misp_j += insts*j_mpki/1000
                        total_misp_i += insts*i_mpki/1000
                        total_misp_c += insts*c_mpki/1000
                        total_misp_r += insts*r_mpki/1000
                        total_rdc_sc += insts*sc_rdc_mpki/1000
                    coverage += weight
                    count += 1
            # print(1111)
            # print(tree)
            # print(bmk_stat)
            if merge_benckmark:
                bmk_stat[conf][bmk] = {}
                bmk_stat[conf][bmk]['time'] = time
                ref_time = c.get_spec_ref_time(bmk, ver)
                assert ref_time is not None
                bmk_stat[conf][bmk]['ref_time'] = ref_time
                bmk_stat[conf][bmk]['score'] = ref_time / time
                bmk_stat[conf][bmk]['Coverage'] = coverage/count
                bmk_stat[conf][bmk]['IPC'] = total_inst / total_cycle
                bmk_stat[conf][bmk]['mpki_b'] = 1000 * total_misp_b / total_inst
                bmk_stat[conf][bmk]['misrate_b'] = 100 * total_misp_b / total_branches

                if prefix == 'xs_':
                    bmk_stat[conf][bmk]['mpki'] = 1000 * total_misp / total_inst
                    # bmk_stat[conf][bmk]['mpki_b_ubtb'] = 1000 * total_misp_b_ubtb / total_inst
                    # bmk_stat[conf][bmk]['mpki_b_btb'] = 1000 * total_misp_b_btb / total_inst
                    bmk_stat[conf][bmk]['mpki_j'] = 1000 * total_misp_j / total_inst
                    bmk_stat[conf][bmk]['mpki_i'] = 1000 * total_misp_i / total_inst
                    bmk_stat[conf][bmk]['mpki_c'] = 1000 * total_misp_c / total_inst
                    bmk_stat[conf][bmk]['mpki_r'] = 1000 * total_misp_r / total_inst
                    bmk_stat[conf][bmk]['mpki_rdc_sc'] = 1000 * total_rdc_sc / total_inst

    for conf in confs:
        print(conf, '='*60)
        df = pd.DataFrame.from_dict(workload_dict[conf], orient='index')
        workload_dict[conf] = df
        # pd.set_option('display.max_columns', None)
        print(df)

        if merge_benckmark:
            df = pd.DataFrame.from_dict(bmk_stat[conf], orient='index')
            bmk_stat[conf] = df
            excluded = df[df['Coverage'] <= min_coverage]
            df = df[df['Coverage'] > min_coverage]
            
            # print(df)
            print(df.sort_index())
            df.to_csv('spec06_'+conf+'.csv')
            print(f'Estimated score @ {clock_rate/(10**9)}GHz:', geometric_mean(df['score']))
            print('Estimated score per GHz:', geometric_mean(df['score'])/(clock_rate/(10**9)))
            print('Excluded because of low coverage:', list(excluded.index))


    tests = []
    for conf in confs.keys():
        if conf == base:
            continue
        rel = workload_dict[conf]['IPC']/workload_dict[base]['IPC']
        print(f'{conf}  Mean relative performance: {gmean(rel)}')
        tests.append(rel)
    if len(tests):
        dfx = pd.concat(tests, axis=1)
        print('Relative performance:')
        print(dfx)


def gem5_spec2006():
    ver = '06'
    confs = {
            # 'O1': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH1S1G1Config',
            # 'O2': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH1S1G2CL0Config',
            # 'O1S0': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH1S0G1Config',
            # 'O1X': '/home51/zyy/expri_results/omegaflow_spec17/XOmegaH1S1G1Config',
            # 'O1*-': '/home51/zyy/expri_results/omegaflow_spec17/OmegaH0S0G1Config',
            # 'F1-': '/home51/zyy/expri_results/omegaflow_spec17/FFS0Config',
            # 'F1H': '/home51/zyy/expri_results/omegaflow_spec17/FFH1Config',
            # 'GEM5': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-04-09_22:21:46'
            'GEM5_LTAGE': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-10-12_21:52:21',
            'GEM5_TAGE': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-09-30_15:46:10',
            'GEM5_Tour': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-10-12_21:31:41',
            'GEM5_BiMode': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-10-12_21:37:50',
            # 'GEM5hist128': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-06-23_14:08:12',
            # 'GEM5hist256': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-06-23_14:25:29',
            # 'GEM5hist512': '/home51/glr/expri_results/SPEC06/FullWindowO3Config_2021-06-23_14:55:46',
            # 'F2': '/home51/zyy/expri_results/omegaflow_spec17/FFG2CL0CG1Config',
            # 'FullO3': '/home51/zyy/expri_results/omegaflow_spec17/FullWindowO3Config'
            }

    compute_weighted_mpki(
            ver=ver,
            confs=confs,
            base='GEM5_TAGE',
            simpoints=f'/home/glr/gem5_data_proc/simpoint_coverage0.3.json',
            prefix = '',
            stat_file='m5out/stats.txt',
            insts_file_fmt =
            '/bigdata/zzf/spec_cpt/logs/profiling/{}.log',
            clock_rate = 4 * 10**9,
            min_coverage = 0.1,
            # blacklist = ['gamess'],
            whitelist = ['hmmer'],
            merge_benckmark=True,
            )


def xiangshan_spec2006():
    ver = '06'
    confs = {
            # 'XiangShan_decoupled': '/home/glr/spec_output/xs_simpoint_batch/SPEC06_EmuTasksConfig_2021-09-03',
            # 'XiangShan_decoupled': '/home53/glr/spec_output/xs_simpoint_batch/SPEC06_EmuTasksConfig_2021-09-06',
            # 'XiangShan_his128': '/home53/glr/spec_output/xs_simpoint_batch/SPEC06_EmuTasksConfig_hist128_2021-10-05',
            'XiangShan_his64': '/home53/glr/spec_output/xs_simpoint_batch/SPEC06_EmuTasksConfig_hist64_2021-10-06',
            'XiangShan_br1': '/home53/glr/spec_output/xs_simpoint_batch/SPEC06_EmuTasksConfig_br1_2021-10-08',
            # 'XiangShan_master_fp_0.3': '/home/ljw/master-40-perf/output',
            # 'XiangShan2': '/home51/glr/expri_results/xs_simpoint_batch/emu-master-4t/SPEC06_EmuTasksConfig'
            # 'XiangShan2': '/home/zyy/expri_results/xs_simpoint_batch/SPEC06_EmuTasksConfig-04-19-2021',
            }
    int_list = [
        "perlbench", "bzip2", "gcc", "mcf", 
        "gobmk", "hmmer", "sjeng", "libquantum",
        "h264ref", "omnetpp", "astar", "xalancbmk"
        ]
    bp_list = [
        "perlbench", "povray", "astar", "gcc",
        "omnetpp", "mcf", "soplex", "bzip2",
        "namd", "sjeng", "hmmer", "xalancbmk", "sphinx3"
    ]
    fp_list = []
    compute_weighted_mpki(
            ver=ver,
            confs=confs,
            # base='XiangShan_master_fp_0.3',
            base='XiangShan_his64',
            simpoints=f'/home/glr/gem5_data_proc/simpoint_coverage0.3.json',
            prefix = 'xs_',
            stat_file='main_err.txt',
            insts_file_fmt =
            '/bigdata/zzf/spec_cpt/logs/profiling/{}.log',
            clock_rate = 2 * 10**9,
            min_coverage = 0.1,
            # blacklist = int_list,
            # whitelist = int_list,
            # whitelist = bp_list,
            # whitelist = ['gobmk'],
            # blacklist = ['calculix'],
            merge_benckmark=True,
            )



if __name__ == '__main__':
    xiangshan_spec2006()
    # gem5_spec2006()
