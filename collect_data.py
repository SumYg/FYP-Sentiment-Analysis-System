from common.file_handler import read_parquet, save_df2parquet, current_time_string, save2csv
from common.twitter_api import TwitterAPI
from common.google_trends import GoogleTrends
import logging
import sys
from os import makedirs
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

def pipeline(tweets_no):
    google_trends = GoogleTrends()

    keywords_file_map = []

    # skip = True

    for _, keyword in google_trends.get_trending_searches().itertuples():
        logging.info("Going to get "+keyword)

        # if keyword == 'Boise State football':
        #     skip = False
        
        # if skip:
        #     continue
        
        # keyword = 'Hurricane tracker'
        try:
            twitter_api = TwitterAPI()
            result = twitter_api.search_tweet_by_keyword(keyword, tweets_no=tweets_no)
            print(result)
            safe_name = sanitize_filepath(keyword)
            save_path = save_df2parquet(result, safe_name)
            keywords_file_map.append((keyword, save_path))
        except Exception as E:
            logging.error(E)

    save2csv(pd.DataFrame(data=keywords_file_map, columns=('Keyword', 'Filename')), 'keyword2file')

if __name__ == '__main__':
    # pipeline(tweets_no=1000)

    logging.info("Load parquet from disk")
    df = read_parquet('./data\China_2022-09-25T03-08-42.parquet.bz')
    print(df)

    # print("Read from local")
    # df = read_parquet(saved_path)
    # print(df)
