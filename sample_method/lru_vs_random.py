import os
import os.path as osp
import utils as u
import utils.common as c
import pandas as pd


top_dir = '/home51/zyy/expri_results'
configs = {
    'lru': osp.join(top_dir, 'gem5_ooo_spec06_shotgun', 'gcc_200'),
    'random': osp.join(top_dir, 'gem5_ooo_spec06_shotgun_rand_icache', 'gcc_200'),
}


def overall():
    dfs = {}
    for conf, path in configs.items():
        stat_files = u.glob_stats(path)
        ipcs = {}
        for i, f in stat_files:
            ipcs[int(i)] = u.get_ipc(f)
        df = pd.DataFrame.from_dict(ipcs, orient='index')
        print(conf)
        print(df.values.mean())
        dfs[conf] = df
    return dfs


def simpoint_sample(dfs):
    results = {}
    for i in range(0, 100):
        # id_map_file = osp.join('/home51/zyy/projects/betapoint/models', f'simpoint_class_map_{i}.txt')
        # id_map_file = osp.join('/home51/zyy/projects/betapoint/simpoint_class_maps', f'simpoint_class_map_dr_{i}_k15.txt')
        id_map_file = osp.join('/home51/zyy/projects/betapoint/simpoint_class_maps', f'simpoint_class_map_dr_{i}.txt')
        ipc_map = {}
        for config, df in dfs.items():
            ipcs = u.class_map_to_ipc_df(df, id_map_file)
            print(config)
            print(ipcs.values.mean())
            ipc_map[config] = ipcs.values.mean()
        benefit = (ipc_map["lru"] - ipc_map["random"]) / ipc_map["random"]
        print(f'(lru - rand)/rand = {benefit}')
        results[i] = benefit
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv('k30_lru_vs_random.csv')


if __name__ == '__main__':
    dfs = overall()
    simpoint_sample(dfs)
