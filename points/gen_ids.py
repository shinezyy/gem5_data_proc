import os
import os.path as osp
import pandas as pd


def gen_ids(input_file, output_file):
    df = pd.read_csv(input_file)
    print(df)
    with open(output_file, 'w') as f:
        for i, values in df.iterrows():
            print(int(values[0]), file=f)


if __name__ == '__main__':
    gen_ids('outputs/gcc_200/shotgun_ipc.csv', 'outputs/gcc_200/shotgun_idmap.txt')
