import pandas as pd


def class_map_to_ipc_df(real_ipc_df, path):
    df2 = {}
    with open(path) as f:
        for line in f:
            k, v = line.strip().split(' ')
            df2[int(k) - 1] = real_ipc_df.loc[int(v) - 1][0]
    df2 = pd.DataFrame.from_dict(df2, orient='index')
    df2 = df2.sort_index(ascending=True)
    return df2