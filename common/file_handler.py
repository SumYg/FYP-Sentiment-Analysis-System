from pathvalidate import sanitize_filename
from os import makedirs
from os.path import join as join_paths
import pandas as pd
import datetime
import logging

ROOT_PATH = './data'

def current_time_string():
    ft = "%Y-%m-%dT%H-%M-%S"
    t = datetime.datetime.now().strftime(ft)
    return t

class FilesSaver:
    def __init__(self) -> None:
        self.files_path = join_paths(ROOT_PATH, current_time_string())
        makedirs(self.files_path)

    def save_df2parquet(self, df, filename, path=None, check_name=True):
        if check_name:
            filename = sanitize_filename(filename)
        save_path = join_paths(path if path else self.files_path, f"{filename}_{current_time_string()}.parquet.br")
        logging.info(f"Save to {save_path}")
        df.to_parquet(save_path, compression='brotli')
        logging.info(f"Saved df to {save_path}")
        return save_path

    def save2csv(self, df, filename, path=None, check_name=True):
        if check_name:
            filename = sanitize_filename(filename)
        save_path = join_paths(path if path else self.files_path, f"{filename}.csv")
        # save_path = join_paths(path if path else self.files_path, f"{filename}_{current_time_string()}.csv")
        df.to_csv(save_path, index=False)

def read_parquet(path):
    logging.info(f"Read df from {path}")
    return pd.read_parquet(path)





# split parquet by size
# https://stackoverflow.com/questions/59887234/split-a-parquet-file-in-smaller-chunks-using-dask

if __name__ == '__main__':
    makedirs(ROOT_PATH, exist_ok=True)

    # import pandas as pd
    # df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
    # print(save_df2parquet(df, 'testing'))

    df = read_parquet(r'./data\2022-09-25T11-46-32/Albert Pujols_2022-09-25T12-29-55.parquet.bz')
    print(df)
    # print(current_time_string())
