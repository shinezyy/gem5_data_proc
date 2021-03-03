import pandas as pd
import os.path as osp


def sample(fname: str, n_samples):
    assert osp.isfile(fname)
    df = pd.read_csv(fname, header=0, index_col=0)
    df = df.sample(n_samples)
    df.sort_index(inplace=True)
    df = df['ipc']
    return df


if __name__ == '__main__':
    for workload in ['bzip2_liberty', 'gcc_200', 'gobmk_13x13',
            'mcf',  'perlbench_diffmail',  'xalancbmk',]:
        df = sample(f'/home51/zyy/projects/gem5_data_proc/outputs/shotgun_continuous_point/{workload}.csv', 300)
        df.to_csv(f'/home51/zyy/projects/gem5_data_proc/outputs/shotgun_continuous_point/{workload}_sampled.csv')
