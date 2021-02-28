import utils as u
import utils.common as c
import utils.target_stats as t
import json
import pandas as pd
import numpy as np
from scipy.stats import gmean

simpoints = '/home51/zyy/expri_results/simpoints.json'

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


def main():
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
    stat_dict = {}
    for conf, file_path in confs.items():
        stat_dict[conf] = {}
        tree = u.glob_weighted_stats(
                file_path,
                u.single_stat_factory(t.ipc_target, 'cpi')
                )
        with open(simpoints) as jf:
            js = json.load(jf)
        for bmk in tree:
            cpis = []
            for workload, df in tree[bmk].items():
                # filter out small points
                selected = dict(js[workload])
                keys = [int(x) for x in selected]
                df = df.loc[keys]
                cpi = u.weighted_cpi(df)
                print(cpi)
                cpis.append(cpi)
                # stat_dict[conf][workload] = np.mean(cpis)
            stat_dict[conf][bmk] = np.mean(cpis)
    df = pd.DataFrame.from_dict(stat_dict)
    print(df)
    base = 'FullO3'
    tests = []
    for conf in confs.keys():
        if conf == base:
            continue
        rel = df[base]/df[conf]
        print(f'{conf}  Mean relative performance: {gmean(rel)}')
        tests.append(rel)
    dfx = pd.concat(tests, axis=1)
    print(dfx)
    print(dfx[0] - dfx[1])


if __name__ == '__main__':
    main()

