from common.file_handler import read_parquet, current_time_string, FilesSaver
# from common.twitter_api import TwitterAPI
from common.twitter_api import sn_search_within_day
from common.reddit_api import RedditAPI
from common.google_trends import GoogleTrends
import logging
import sys
from os import makedirs
from os.path import basename
import pandas as pd
from traceback import format_exc

from grouping import process

from common.sql_db import MyDB
import datetime

from data_cleaning import preprocess
import spacy

makedirs('./log', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"./log/{current_time_string()}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def pipeline(target_tweets_no=1000):
    google_trends = GoogleTrends()

    keywords_file_map = []

    exclude_set = {}
    # exclude_set = {'Broncos', 'Mario Movie', 'Omonia vs Man United', 'Pixel 7', 'Thailand'}
    # skip = True
    # twitter_api = TwitterAPI()
    reddit_api = RedditAPI()
    files_saver = FilesSaver()

    google_trends_keywords = google_trends.get_trending_searches()

    keywords_no = google_trends_keywords.shape[0]

    logging.info(f"Target Number of Tweets: {target_tweets_no}")
    avg_tweets_no = target_tweets_no // keywords_no
    logging.info(f"Average Number of Tweets per keyword: {avg_tweets_no}")
    assert avg_tweets_no >= 10, "At least 10 tweets in each API call"

    keywords_no += 1
    all_posts = []

    
    current_date = datetime.date.today()

    nlp = spacy.load("en_core_web_sm")

    try:
        for i, keyword in google_trends_keywords[:10][::-1].itertuples():
            keywords_no -= 1
            if keyword in exclude_set:
                continue
            # related = google_trends.get_suggestions(keyword)
            # logging.info(keyword)
            # if related:
            #     cont = True
            #     logging.info(related)
            #     for k in related:
            #         title = k['title']
            #         if title != keyword:
            #             keyword = title
            #             cont = False
            #             break
            #     if cont:
            #         continue

            # else:
            #     continue
            all_posts = []
            tweets_no = target_tweets_no// keywords_no

            logging.info("Going to get "+keyword)

            reddit_submissions, reddit_comments = reddit_api.search_keyword(keyword)
            all_posts.append(reddit_submissions[['text', 'ups']])
            all_posts.append(reddit_comments[['body', 'ups']])
            reddit_submissions_no, reddit_comments_no = reddit_submissions.shape[0], reddit_comments.shape[0]
            logging.info(f"Got {reddit_submissions_no} Submissions, {reddit_comments_no} Comments from Reddit api")
            

            reddit_submissions_saved_name = files_saver.save_df2parquet(reddit_submissions, f"{keyword}_reddit_submissions")
            reddit_comments_saved_name = files_saver.save_df2parquet(reddit_comments, f"{keyword}_reddit_comments")

            result = sn_search_within_day(keyword, tweets_no=tweets_no)
            all_posts.append(result[['text', 'like_count']])
            # print(result)
            result_no = result.shape[0]
            if result_no < tweets_no:
                logging.warning("Not enough tweets collected: "
                                f"Expected {tweets_no} but got {result_no} tweets")
            target_tweets_no -= result_no
            
            twitter_file_saved_name = files_saver.save_df2parquet(result, f"{keyword}_twitter")

            keywords_file_map.append((keyword, basename(twitter_file_saved_name), result_no
                , basename(reddit_submissions_saved_name), reddit_submissions_no
                , basename(reddit_comments_saved_name), reddit_comments_no))
            
            for posts_df in all_posts:
                posts_df.columns = ['text', 'likes']
            process_posts(pd.concat(all_posts, axis=0, ignore_index=True), current_date, keyword, i, nlp)
        
            # break
            # return
    except Exception as E:
            logging.error(f"{type(E)}\n"
                          f"{format_exc()}")
        

    files_saver.save2csv(pd.DataFrame(data=reversed(keywords_file_map), columns=('Keyword', 'Twitter File', 'Tweets Count'
                                                                        , 'Reddit Submissions File', 'Reddit Submissions Count'
                                                                        , 'Reddit Comments File', 'Reddit Comments Count')), 'keyword2file')


class SplitedPost:
    def __init__(self, preprocessed_text, posts, nlp):
        text = []
        post_id = []
        for i, t in enumerate(preprocessed_text):
            doc = nlp(t)
            for splited in doc.sents:
                text.append(splited.text.strip())
                post_id.append(i)

        self.text = text
        self.post_id = post_id
        self.posts = posts
    
    def __getitem__(self, index):
        return self.text[index], self.posts[self.post_id[index]][1], self.post_id[index]
    
    def __len__(self):
        return len(self.posts)


def process_posts(posts, current_date, keyword, i, nlp):
    """
    posts: df with columns ['text', 'likes']
    """
    logging.info(f"Drop duplicate posts, original no. of posts: {len(posts)}")
    posts.drop_duplicates(subset=['text'], inplace=True)

    logging.info(f"Preprocessing post with length {len(posts)}")
    posts['text'] = posts['text'].apply(preprocess)

    logging.info(f"Drop duplicate posts after preprocessing, no. of posts: {len(posts)}")
    posts.drop_duplicates(subset=['text'], inplace=True)


    logging.info(f"Removing empty posts")
    posts = posts[posts['text'].apply(lambda x: len(x) > 0)]
        
    text = posts['text'].tolist()
    posts = posts.to_numpy().tolist()

    logging.info(f"Splitting posts")
    splited_posts = SplitedPost(text, posts, nlp)
    logging.info(f"No. of splited posts: {len(splited_posts.text)}")

        
    semtiment, similar_opinions_return, entailed_opinions_return = process(splited_posts, splited_posts.text)

    db = MyDB()
    db.insert_keywords([[current_date, keyword, semtiment, len(splited_posts), i]])
    

    db.insert_opinions(similar_opinions_return, i)
    db.insert_opinions(entailed_opinions_return, i, class_=1)

if __name__ == '__main__':
    pipeline(300000)

    # logging.info("Load parquet from disk")
    # df = read_parquet('./data\Clemson football_2022-09-25T11-28-15.parquet.bz')
    # print(df)

    # print("Read from local")
    # df = read_parquet(saved_path)
    # print(df)


    # text = [p[0] for p in post]
    # g = OpinionGrouper('bin/2023-Apr-08-10:29:46/E24.pytorch', batch_size=128, score_threshold=0.6)
    # print(g.get_unordred_pairs_score(text))
