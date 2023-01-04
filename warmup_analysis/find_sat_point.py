import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import math as m


def saturated_at(baseline_mpki, current_mpki, abs_thres):
    if baseline_mpki < 0.001:
        relative_sat = False
    else:
        relative_sat = current_mpki/baseline_mpki < 1.03
    abs_sat = current_mpki - baseline_mpki < abs_thres
    # if abs_sat:
    #     print(f'baseline_mpki: {baseline_mpki}, current_mpki: {current_mpki}, abs_thres: {abs_thres}')
    return relative_sat or abs_sat

def find_sat_points(df, tag):
    workloads = {}
    max_len = 95
    for instance in list(df.index):
        print(instance)
        workload_warmup = instance.split('-')
        workload = workload_warmup[0]
        skip, fw, dw, sample = workload_warmup[1:]
        # print(workload, fw, dw, sample)
        if workload in workloads:
            workloads[workload].append((fw, dw, sample, instance))
        else:
            workloads[workload] = [(fw, dw, sample, instance)]
    selected = []
    to_drop = []
    for k, v in workloads.items():
        # print(len(v))
        if len(v) != 9:
            print(f'Exclude {k}')
            to_drop.append(k)
            for row in v:
                print(f'Drop {row[3]}')
                df = df.drop(row[3], inplace=False)
    for k in to_drop:
        workloads.pop(k)

    print(f'After drop, df shape: {df.shape}, df samples: {df.shape[0]/9}')
    saturation = {}
    new_df = {}
    for k in workloads:
        baseline = {}
        for dw in [max_len, 50, 25, 10, 5, 4, 3, 2, 1]:
            instance = f'{k}-{max_len-dw}-{0}-{dw}-5'
            if dw == max_len:
                saturation[k] = {}
                baseline['ipc'] = df.loc[instance, 'ipc']
                baseline['bmpki'] = df.loc[instance, 'total branch MPKI']
                baseline['l3mpki'] = df.loc[instance, 'L3MPKI']
                baseline['l2mpki'] = df.loc[instance, 'L2MPKI']

            if saturated_at(baseline['bmpki'], df.loc[instance, 'total branch MPKI'],
                            abs_thres=((1000/200)/(baseline['ipc'] * 30))):
                saturation[k]['bmpki'] = (
                    dw, baseline['bmpki'], df.loc[instance, 'total branch MPKI'])

            if saturated_at(baseline['l3mpki'], df.loc[instance, 'L3MPKI'],
                            abs_thres=((1000/200)/(baseline['ipc'] * 250))):
                saturation[k]['l3mpki'] = (
                    dw, baseline['l3mpki'], df.loc[instance, 'L3MPKI'])
        print(k, saturation[k])
        assert len(saturation[k]) == 2
        new_df[k] = *saturation[k]['bmpki'], *saturation[k]['l3mpki']
    new_df = pd.DataFrame.from_dict(new_df, orient='index', columns=[
                                    'br_dw', 'bmpki_baseline', 'bmpki_sat', 'l3_dw', 'l3mpki_baseline', 'l3mpki_sat'])
            
    new_df['max_demand'] = new_df[['br_dw', 'l3_dw']].max(axis=1)

    new_df.to_csv(osp.join('results', f'{tag}-5M-warmup-demand.csv'))
    # print(new_df)

    full_demand = {}
    for demand in (max_len, 50, 25, 10, 5, 4, 3, 2, 1):
        full_demand[demand] = new_df[new_df['max_demand'] == demand].shape[0]
        print(f'{demand}: {new_df[new_df["max_demand"] == demand].shape[0]}')
    full_demand = pd.DataFrame.from_dict(full_demand, orient='index', columns=['count'])
    full_demand.to_csv(osp.join('results', f'{tag}-5M-full-demand-dist.csv'))
    print(full_demand)

    br_demand = {}
    for demand in (max_len, 50, 25, 10, 5, 4, 3, 2, 1):
        br_demand[demand] = new_df[new_df['br_dw'] == demand].shape[0]
        print(f'{demand}: {new_df[new_df["br_dw"] == demand].shape[0]}')
    br_demand = pd.DataFrame.from_dict(br_demand, orient='index', columns=['count'])
    br_demand.to_csv(osp.join('results', f'{tag}-5M-br-demand-dist.csv'))
    print(br_demand)


def draw_point(df: pd.DataFrame, point: str, axs, fig, tag, total_plots=4, current_plot=0, font=''):
    max_len = 95
    configs = []
    dws = np.array([max_len, 50, 25, 10, 5, 4, 3, 2, 1])
    for dw in dws:
        configs.append(f'{point}-{max_len-dw}-{0}-{dw}-5')
    # print(df.columns)
    # print(df.loc[configs, 'total branch MPKI'])

    if axs is None:
        fig, axs = plt.subplots((total_plots + 1) // 2, 2, sharex=True, sharey=True)
    ax = axs[current_plot // 2, current_plot % 2]
    ax.plot(dws, df.loc[configs, 'total branch MPKI'], label=tag)
    short_tag = '_'.join(point.split('_')[:-1])
    ax.set_title(short_tag, font=font, style='normal')
    
    
    return axs, fig



def gen_sat_curve_csv():
    # top_dir = '/nfs-nvme/home/zhouyaoyang/gem5-results'
    # warmup_results_list = {'gem5': ['gem5-sat-point-low.csv', 'gem5-sat-point-25M.csv',
    #                                 'gem5-sat-point-50M.csv', 'gem5-sat-point-95M.csv', ],
    #                        'xs': ['xs-vs-gem5-sat-point-low.csv', 'xs-vs-gem5-sat-point-25M.csv',
    #                               'xs-vs-gem5-sat-point-50M.csv', 'xs-vs-gem5-sat-point-95M.csv', ]}
    # ax = None
    # fig = None
    # for k, warmup_results in warmup_results_list.items():
    #     dfs = []
    #     for csv in warmup_results:
    #         df = pd.read_csv(osp.join(top_dir, csv), index_col=0)
    #         df.drop(df.tail(1).index, inplace=True)
    #         dfs.append(df)
    #         print('Shape:', df.shape)
    #         # print(df)
    #     merged = pd.concat(dfs, axis=0)
    #     dedup = merged[~merged.index.duplicated(keep='first')]

    #     # find_sat_points(dedup, k)
    #     ax, fig = draw_point(dedup, 'h264ref_foreman_401300000000', ax, fig, k)
    # plt.show()

    top_dir = '/nfs-nvme/home/zhouyaoyang/gem5-results'
    warmup_results_list = {
        'gem5': [
            'gem5-sat-point-95M.csv',
            'gem5-sat-point-50M.csv',
            'gem5-sat-point-25M.csv',
            'gem5-sat-point-low.csv',
        ],
        'xs': [
            'xs-vs-gem5-sat-point-95M.csv',
            'xs-vs-gem5-sat-point-50M.csv',
            'xs-vs-gem5-sat-point-25M.csv',
            'xs-vs-gem5-sat-point-low.csv',
        ],
    }
    max_len = 95
    point_dfs = []
    for cpt in ['perlbench_diffmail_30400000000', 'sjeng_1439300000000',
                'xalancbmk_174600000000', 'gobmk_trevorc_145200000000']:
        br_dfs = []

        for k, warmup_results in warmup_results_list.items():
            dfs = []
            for csv in warmup_results:
                df = pd.read_csv(osp.join(top_dir, csv), index_col=0)
                df = df.drop(df.tail(1).index)
                dfs.append(df)
                # print('Shape:', df.shape)
                # print(df)
            merged = pd.concat(dfs, axis=0)
            dedup = merged[~merged.index.duplicated(keep='first')]

            dws = np.array([max_len, 50, 25, 10, 5, 4, 3, 2, 1])
            configs = []
            for dw in dws:
                configs.append(f'{cpt}-{max_len-dw}-{0}-{dw}-5')
            
            br_slice = dedup.loc[configs, ['total branch MPKI']]
            br_slice.columns = [f'{k}']
            br_slice['dw'] = dws
            br_slice = br_slice.sort_values(by='dw')
            # print(br_slice)
            br_dfs.append(br_slice)
        df = pd.concat(br_dfs, axis=1)
        # print(df)
        point_dfs.append(df)
    point_dfs = pd.concat(point_dfs, axis=0)
    print(point_dfs)
    point_dfs.to_csv(osp.join('results', f'br-mpki-sat-curve.csv'))

                # point = '_'.join(cpt.split('_')[:-1])
                # df['point'] = [point] * df.shape[0]

            # find_sat_points(dedup, k)
            # ax, fig = fsp.draw_point(dedup, cpt, ax, fig, k, 4, cur_plot, eng_font)
    # print(dfs)

def get_func_warmup_rank():
    top_dir = '/nfs-nvme/home/zhouyaoyang/gem5-results'
    warmup_results_list = {
        'gem5': [
            'gem5-sat-point-95M.csv',
            'gem5-sat-point-50M.csv',
            'gem5-sat-point-25M.csv',
            'gem5-sat-point-low.csv',
        ],
        'xs': [
            'xs-90M-5M-5M-used.csv',
            'xs-vs-gem5-sat-point-95M.csv',
            'xs-vs-gem5-sat-point-50M.csv',
            'xs-vs-gem5-sat-point-25M.csv',
            'xs-vs-gem5-sat-point-low.csv',
        ],
    }
    dfs = []
    for csv in warmup_results_list['xs']:
        df = pd.read_csv(osp.join(top_dir, csv), index_col=0)
        df = df.drop(df.tail(1).index)
        dfs.append(df)
        # print('Shape:', df.shape)
        # print(df)
    merged = pd.concat(dfs, axis=0)

    warmup_lengths = merged.index.str.split('-')
    merged.loc[:, 'fw'] = warmup_lengths.str[2].astype(int)
    merged.loc[:, 'dw'] = warmup_lengths.str[3].astype(int)
    merged['warmup'] = merged['dw'] + merged['fw']
    dedup = merged[~merged.index.duplicated(keep='first')]
    # print(dedup)
    workloads = [x.split('-')[0] for x in list(dedup.index)]
    for workload in sorted(list(set(workloads))):
        wl_results = []
        for x in list(dedup.index):
            if x.startswith(workload):
                wl_results.append(x)
        # print(wl_results)
        wl_df = dedup.loc[wl_results, ['L2MPKI', 'L3MPKI', 'ipc', 'fw', 'dw', 'warmup']]

        wl_df.sort_values(by='L2MPKI', inplace=True)

        long_warmup_rank = (wl_df.index.get_loc(
            f'{workload}-0-0-95-5') + wl_df.index.get_loc(f'{workload}-45-0-50-5'))/2
        pure_dw_df = wl_df[~wl_df.index.str.contains('-90-5-')]
        correlation = pure_dw_df['L2MPKI'].corr(pure_dw_df['warmup'])
        # print(correlation)
        if correlation < -0.5:
            # print(workload, correlation)
            # print(wl_df)
            # print(pure_dw_df)
            func_warm_rank = wl_df.index.get_loc(f'{workload}-0-90-5-5')
            print('func warmup rank:', func_warm_rank, end=', ')
            print('High warmup rank:', long_warmup_rank)
            if func_warm_rank >= 4:
                print(wl_df)



if __name__ == '__main__':
    # gen_sat_curve_csv()
    get_func_warmup_rank()
