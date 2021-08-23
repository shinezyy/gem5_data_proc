import pandas as pd
import os.path as osp

base_out_dir = '/home51/zyy/projects/gem5_data_proc/outputs/shotgun_continuous_point'
base_pred_dir = '/home51/zyy/predictions/beta_pred'


def sample(fname: str, n_samples):
    print(fname)
    assert osp.isfile(fname)
    df = pd.read_csv(fname, header=0, index_col=0)
    df = df.sample(n_samples)
    df.sort_index(inplace=True)
    # df = df['ipc']
    return df


def sample_plus(fname: str, old_samples, new_samples):
    assert osp.isfile(fname)
    df = pd.read_csv(fname, header=0, index_col=0)
    old_samples = pd.read_csv(old_samples, header=0, index_col=0)
    new_samples = pd.read_csv(new_samples, header=0, index_col=0)
    new_samples = new_samples[new_samples['Similarity'] > 0.075]
    sampled = (set(old_samples.index) | set(new_samples.index)) & set(df.index)
    df = df.loc[sampled]
    df.sort_index(inplace=True)
    # df = df['ipc']
    return df


def rand_sample_all():
    for workload in ['bzip2_liberty', 'gcc_200', 'gobmk_13x13',
                     'mcf',  'perlbench_diffmail',  'xalancbmk',]:
        df = sample(f'{base_out_dir}/{workload}.csv', 300)
        df.to_csv(f'{base_out_dir}/{workload}_sampled.csv')


def sample_suffix(workload, input_suffix, output_suffix):
    df = sample(f'{base_out_dir}/{workload}{input_suffix}.csv', 300)
    df.to_csv(f'{base_out_dir}/{workload}{output_suffix}.csv')


def selective_sample():
    for workload in ['gcc_200']:
        df = sample_plus(
            f'{base_out_dir}/{workload}_smarts8w_mpki.csv',
            f'{base_out_dir}/{workload}_smarts8w_sampled.csv',
            f'/home51/zyy/predictions/beta_pred/gcc_200-smarts8w_large.csv',
        )
        df.to_csv(f'{base_out_dir}/{workload}_smarts8w_sampled_2.csv')


def sample_suffix_all():
    for workload in ['bzip2_liberty', 'gcc_200', 'gobmk_13x13',
                     'perlbench_diffmail',  'xalancbmk',]:
        sample_suffix(workload, '_smarts8w_mpki', '_smarts8w_sampled')


if __name__ == '__main__':
    # sample_suffix_all()
    selective_sample()
