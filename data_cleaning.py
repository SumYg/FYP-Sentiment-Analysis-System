
import re
import emoji
from bs4 import BeautifulSoup
from markdown import markdown


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
    return tweet.strip()

def preprocess(text):
    return cleaner(mark_down2text(text))