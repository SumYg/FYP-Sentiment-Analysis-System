
import dask.dataframe as dd
from glob import glob
import pandas as pd

from bs4 import BeautifulSoup
from markdown import markdown
import re
import emoji
from nltk import sent_tokenize

from common.file_handler import save2pickle

from tqdm import tqdm
from time import time

if __name__ == '__main__':

    PROGRAM_START_TIME = time()

    RATIOS = {'train': .6, 'valid': .2, 'test': .2}

    # MAX_NO = 100000 //3
    # MAX_NO = 200
    MAX_NO = None

    OFFSET = 0

    DATA = []

    def mark_down2text(md):
        html = markdown(md)
        return ''.join(BeautifulSoup(html, "lxml").findAll(text=True)).replace(u'\xa0', u' ')

    def cleaner(tweet):
        tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
        tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
        # tweet = " ".join(tweet.split())
        tweet = emoji.replace_emoji(tweet, replace='')
        # tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI['en']) #Remove Emojis
        tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
        # tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
        #      if w.lower() in words or not w.isalpha())
        return tweet

    def preprocess(text):
        return cleaner(mark_down2text(text))

    def get_file(col):
        tweets_files = []
        for d in filter(lambda d: 'T' in d, glob('data/*')):
            for f in glob(d+"/*.csv"):
                print(f)
                csv = pd.read_csv(f)
                tweets_files.extend(d+'/'+csv[col])
        return tweets_files


    twitter_files = get_file('Twitter File')
    twitter_df = dd.read_parquet(twitter_files)
    # print(type(twitter_df))
    # print(twitter_df)
    # print(twitter_df.head())
    long_text = twitter_df[twitter_df['text'].apply(lambda x: len(x.split())  < 60)]
    print(type(long_text))
    # print(long_text)
    long_text_copy = long_text.drop_duplicates(['tweet_id'], keep='last').drop_duplicates(['text']).reset_index(drop=True)
    # print(len(long_text_copy))
    # print(long_text_copy)
    # print(type(long_text_copy))
    if MAX_NO is not None:
        long_text_copy = long_text_copy.loc[OFFSET:MAX_NO]
    # print(type(long_text_copy))
    # print(len(long_text_copy))
    # for post in tqdm(list(long_text_copy['text'].apply(preprocess))):
    #     DATA.extend(sent_tokenize(post.strip()))

    def prepocess_sentences(df_partition):
        sentences = []
        for post in df_partition['text']:
            sentences.extend(sent_tokenize(preprocess(post).strip()))
        return pd.Series(sentences)

    long_text_copy = long_text_copy.repartition(npartitions=4)
    print(long_text_copy.npartitions)
    DATA.extend(long_text_copy.map_partitions(prepocess_sentences).compute(scheduler='processes', num_workers=4))
    print(f"Time taken to preprocess: {time() - PROGRAM_START_TIME} seconds")
    # save2pickle(tuple(long_text_copy[:MAX_NO]['text'].apply(lambda x: cleaner(mark_down2text(x)))), f'dataset/twitter_{MAX_NO}.pickle')

    # with open("test.data", "w") as f:
    #     for t in long_text_copy['text']:
    #         f.write(t + '\n')
            
    reddit_sub_files = get_file('Reddit Submissions File')
    len(reddit_sub_files)
    twitter_df = dd.read_parquet(reddit_sub_files)
    long_text = twitter_df[twitter_df['text'].apply(lambda x: len(x.split())  < 60)]
    long_text_copy = long_text.drop_duplicates(['reddit_id'], keep='last').drop_duplicates(['text']).reset_index(drop=True)
    # save2pickle(tuple(long_text_copy[:MAX_NO]['text'].apply(lambda x: cleaner(mark_down2text(x)))), f'dataset/reddit_submission_{MAX_NO}.pickle')
    print(len(long_text_copy))
    if MAX_NO is not None:
        long_text_copy = long_text_copy.loc[OFFSET:MAX_NO]
    # for post in tqdm(list(long_text_copy['text'].apply(lambda x: cleaner(mark_down2text(x))).compute(num_workers=4, scheduler='processes'))):
    #     DATA.extend(sent_tokenize(post.strip()))

    long_text_copy = long_text_copy.repartition(npartitions=4)
    print(long_text_copy.npartitions)
    DATA.extend(long_text_copy.map_partitions(prepocess_sentences).compute(scheduler='processes', num_workers=4))
    print(f"Time taken to preprocess: {time() - PROGRAM_START_TIME} seconds")

    reddit_comm_files = get_file('Reddit Comments File')
    twitter_df = dd.read_parquet(reddit_comm_files)
    long_text = twitter_df[twitter_df['body'].apply(lambda x: len(x.split())  < 60)]
    long_text_copy = long_text.drop_duplicates(['comment_id'], keep='last').drop_duplicates(['body']).reset_index(drop=True)
    # save2pickle(tuple(long_text_copy[:MAX_NO]['body'].apply(lambda x: cleaner(mark_down2text(x)))), f'dataset/reddit_comment_{MAX_NO}.pickle')

    print(len(long_text_copy))
    if MAX_NO is not None:
        long_text_copy = long_text_copy.loc[OFFSET:MAX_NO]
    # for post in tqdm(list(long_text_copy['body'].apply(lambda x: cleaner(mark_down2text(x))).compute(num_workers=4, scheduler='processes'))):
    #     DATA.extend(sent_tokenize(post.strip()))
    def prepocess_sentences1(df_partition):
        sentences = []
        for post in df_partition['body']:
            sentences.extend(sent_tokenize(preprocess(post).strip()))
        return pd.Series(sentences)

    long_text_copy = long_text_copy.repartition(npartitions=4)
    print(long_text_copy.npartitions)
    DATA.extend(long_text_copy.map_partitions(prepocess_sentences1).compute(scheduler='processes', num_workers=4))
    print(f"Time taken to preprocess: {time() - PROGRAM_START_TIME} seconds")

    DATA = pd.Series(DATA)
    save2pickle(DATA, f'dataset/sentence_split_full_{len(DATA)}_skip_first_{OFFSET}_all.pickle')
    # change data to list
    DATA = list(DATA)
    # store the file in different splits
    previous = 0
    ratio_items = tuple(RATIOS.items())

    if MAX_NO is None:
        file_name_format = f'dataset/sentence_split_full_{len(DATA)}_skip_first_{OFFSET}'+'.{}.pickle'
    else:
        file_name_format = f'dataset/sentence_split_{len(DATA)}_skip_first_{OFFSET}'+'.{}.pickle'
        # file_name_format = f'dataset/sentence_split_{MAX_NO*3}_skip_first_{OFFSET}'+'.{}.pickle'

    for split_name, split_ratio in ratio_items[:-1]:
        print(split_name)
        split = previous + int(len(DATA) * split_ratio)
        save2pickle(DATA[previous:split], file_name_format.format(split_name))
        previous = split
    save2pickle(DATA[split:], file_name_format.format(ratio_items[-1][0]))
    print(f'Program took {time() - PROGRAM_START_TIME} seconds to run')
