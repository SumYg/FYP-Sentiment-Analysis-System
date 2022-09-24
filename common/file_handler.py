from operator import index
from os import makedirs
from os.path import join as join_paths
import pandas as pd
import datetime
import logging

ROOT_PATH = './data'

def save_df2parquet(df, filename, path=ROOT_PATH):
    save_path = join_paths(path, f"{filename}_{current_time_string()}.parquet.bz")
    df.to_parquet(save_path, compression='brotli')
    logging.info(f"Saved df to {save_path}")
    return save_path

def read_parquet(path):
    logging.info(f"Read df from {path}")
    return pd.read_parquet(path)

def current_time_string():
    #4: to make the time timezone-aware pass timezone to .now()
    ft = "%Y-%m-%dT%H-%M-%S"
    t = datetime.datetime.now().strftime(ft)
    return t

def save2csv(df, filename, path=ROOT_PATH):
    save_path = join_paths(path, f"{filename}_{current_time_string()}.csv")
    df.to_csv(save_path, index=False)


# split parquet by size
# https://stackoverflow.com/questions/59887234/split-a-parquet-file-in-smaller-chunks-using-dask

if __name__ == '__main__':
    makedirs(ROOT_PATH, exist_ok=True)

    # import pandas as pd
    # df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    # print(save_df2parquet(df, 'testing'))

    # df = read_parquet(r'./data\testing.parquet.bz')
    # print(df)
    print(current_time_string())
