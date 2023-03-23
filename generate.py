
import dask.dataframe as dd
from glob import glob
import pandas as pd

from bs4 import BeautifulSoup
from markdown import markdown
import re
import emoji
from nltk import sent_tokenize

from common.file_handler import save2pickle

RATIOS = {'train': .6, 'valid': .2, 'test': .2}

MAX_NO = 30000 //3

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
long_text = twitter_df[twitter_df['text'].apply(lambda x: len(x.split())  < 60)].compute()
long_text_copy = long_text.drop_duplicates(['tweet_id'], keep='last').drop_duplicates(['text']).reset_index(drop=True)
print(len(long_text_copy))
for post in list(long_text_copy[OFFSET:MAX_NO]['text'].apply(lambda x: cleaner(mark_down2text(x)))):
    DATA.extend(sent_tokenize(post.strip()))
# save2pickle(tuple(long_text_copy[:MAX_NO]['text'].apply(lambda x: cleaner(mark_down2text(x)))), f'dataset/twitter_{MAX_NO}.pickle')

# with open("test.data", "w") as f:
#     for t in long_text_copy['text']:
#         f.write(t + '\n')
        
reddit_sub_files = get_file('Reddit Submissions File')
len(reddit_sub_files)
twitter_df = dd.read_parquet(reddit_sub_files)
long_text = twitter_df[twitter_df['text'].apply(lambda x: len(x.split())  < 60)].compute()
long_text_copy = long_text.drop_duplicates(['reddit_id'], keep='last').drop_duplicates(['text']).reset_index(drop=True)
# save2pickle(tuple(long_text_copy[:MAX_NO]['text'].apply(lambda x: cleaner(mark_down2text(x)))), f'dataset/reddit_submission_{MAX_NO}.pickle')
print(len(long_text_copy))
for post in list(long_text_copy[OFFSET:MAX_NO]['text'].apply(lambda x: cleaner(mark_down2text(x)))):
    DATA.extend(sent_tokenize(post.strip()))

reddit_comm_files = get_file('Reddit Comments File')
twitter_df = dd.read_parquet(reddit_comm_files)
long_text = twitter_df[twitter_df['body'].apply(lambda x: len(x.split())  < 60)].compute()
long_text_copy = long_text.drop_duplicates(['comment_id'], keep='last').drop_duplicates(['body']).reset_index(drop=True)
# save2pickle(tuple(long_text_copy[:MAX_NO]['body'].apply(lambda x: cleaner(mark_down2text(x)))), f'dataset/reddit_comment_{MAX_NO}.pickle')

print(len(long_text_copy))
for post in list(long_text_copy[OFFSET:MAX_NO]['body'].apply(lambda x: cleaner(mark_down2text(x)))):
    DATA.extend(sent_tokenize(post.strip()))

# store the file in different splits
previous = 0
ratio_items = tuple(RATIOS.items())
for split_name, split_ratio in ratio_items[:-1]:
    split = previous + int(len(DATA) * split_ratio)
    save2pickle(DATA[previous:split], f'dataset/sentence_split_{MAX_NO*3}_skip_first_{OFFSET}.{split_name}.pickle')
    previous = split
save2pickle(DATA[split:], f'dataset/sentence_split_{MAX_NO*3}_skip_first_{OFFSET}.{ratio_items[-1][0]}.pickle')
