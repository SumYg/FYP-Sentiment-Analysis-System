from common.file_handler import read_parquet, current_time_string, FilesSaver
from common.twitter_api import TwitterAPI
from common.google_trends import GoogleTrends
import logging
import sys
from os import makedirs
from os.path import basename
from pathvalidate import sanitize_filepath
import pandas as pd

makedirs('./log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"./log/{current_time_string()}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def pipeline(tweets_no=100):
    google_trends = GoogleTrends()

    keywords_file_map = []

    # skip = True
    twitter_api = TwitterAPI()

    files_saver = FilesSaver()

    for _, keyword in google_trends.get_trending_searches().itertuples():
        logging.info("Going to get "+keyword)

        # if keyword == 'Laver Cup':
        #     skip = False
        
        # if skip:
        #     continue
        
        # keyword = 'Hurricane tracker'
        try:
            result = twitter_api.search_tweet_by_keyword(keyword, tweets_no=tweets_no)
            print(result)
            safe_name = sanitize_filepath(keyword)
            if result.shape[0] < tweets_no:
                logging.warning("Not enough tweets collected: "
                            f"Expected {tweets_no} but got {result.shape[0]} tweets")
            saved_name = files_saver.save_df2parquet(result, safe_name)
            keywords_file_map.append((keyword, basename(saved_name)))
        except Exception as E:
            logging.error(type(E))
            logging.error(E)
        break

    files_saver.save2csv(pd.DataFrame(data=keywords_file_map, columns=('Keyword', 'Filename')), 'keyword2file')

if __name__ == '__main__':
    # pipeline(tweets_no=10000)
    pipeline()

    # logging.info("Load parquet from disk")
    # df = read_parquet('./data\Clemson football_2022-09-25T11-28-15.parquet.bz')
    # print(df)

    # print("Read from local")
    # df = read_parquet(saved_path)
    # print(df)
