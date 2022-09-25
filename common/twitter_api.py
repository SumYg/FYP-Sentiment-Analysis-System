import tweepy
from tweepy.errors import TooManyRequests
from pprint import pprint
import pandas as pd
import logging
from time import sleep

def retry_when_rate_limit_exceed(func):
    def inner(*args, **kwargs):
        retry_interval = 100
        max_trials = 1000  # avoid infinite loop
        trials = 0
        while trials < max_trials:
            try:
                return func(*args, **kwargs)
            except TooManyRequests:
                """
                May examine the HTTP header for more precise sleep time
                Recovering from a rate limit
                https://developer.twitter.com/en/docs/twitter-api/rate-limits
                """
                trials += 1
                logging.info(f"Too Many Requests, Retry after {retry_interval}s, current trial: {trials}")
                sleep(retry_interval)
                retry_interval += 5
            except Exception as E:
                # from collections import namedtuple
                # Empty = namedtuple('Empty', 'data', 'meta')
                logging.error(f"{type(E)}{E}\n"
                                "End this keyword search")
                # return Empty([], {})
                return None
        raise

    return inner
    

class TwitterAPI:
    def __init__(self) -> None:
        with open('./cred/twitter.txt', 'r') as f:
            bearer_token = f.read()
        print(f"Token: {bearer_token}")
        self.client = tweepy.Client(bearer_token=bearer_token)
        # client.get_bookmarks()
        # auth = tweepy.OAuth2BearerHandler(bearer_token)  # App only
        # self.api = tweepy.API(auth)
        # public_tweets = api.home_timeline()
        # for tweet in public_tweets:
        #     print(tweet.text)

    # @retry_when_rate_limit_exceed
    def search_tweet_by_keyword(self, keyword, tweets_no=100):

        def parse_tweet(tweet):
            pm = tweet.public_metrics
            data.append((tweet.id, tweet.text, tweet.created_at, pm['retweet_count'], pm['reply_count'], pm['like_count'], pm['quote_count']))

        @retry_when_rate_limit_exceed
        def get_tweets(*args, **kwargs):
            return self.client.search_recent_tweets(*args, **kwargs)
            
        """
        Ref: https://dev.to/twitterdev/a-comprehensive-guide-for-using-the-twitter-api-v2-using-tweepy-in-python-15d9
        https://developer.twitter.com/en/docs/twitter-api/fields
        """
        # return self.client.search_recent_tweets(query=keyword, tweet_fields=['context_annotations', 'created_at', 'public_metrics'], max_results=100)
        data = []
        columns = ('tweet_id', 'text', 'created_at', 'retweet_count', 'reply_count', 'like_count', 'quote_count')
        
        if tweets_no <= 100:
            logging.info("Request Twitter API")
            tweets = get_tweets(query=f"{keyword} -is:retweet lang:en", tweet_fields=['created_at', 'public_metrics'], max_results=tweets_no)
            logging.info("Request Twitter API Finished")
            for tweet in tweets.data:
                parse_tweet(tweet)
        else:
            logging.info("Request Twitter API for multiple batches")
            next_token = None
            for i in range(tweets_no // 100):
                start = i * 100
                end = (i + 1)* 100
                if end > tweets_no:
                    end = tweets_no
                current_no = end - start

                tweets = get_tweets(query=f"{keyword} -is:retweet lang:en", tweet_fields=['created_at', 'public_metrics'], max_results=current_no, next_token=next_token)
                logging.info("Request Twitter API Finished")
                if tweets and tweets.data:
                    for tweet in tweets.data:
                        parse_tweet(tweet)
                    if 'next_token' not in tweets.meta:
                        logging.warning(f"Don't have next token for keyword: {keyword}")
                        break
                    next_token = tweets.meta['next_token']
                else:
                    logging.warning(f"Early Break: {keyword}")
                    break
            # print("--------")
            
            # print(tweet.id)
            # pprint(tweet.text)
            # if len(tweet.context_annotations) > 0:
            #     print("=====")
            #     print(tweet.context_annotations)
            #     print("=====")
            # print("Context_annotations", tweet.context_annotations)
            # print('Created At', tweet.created_at)
            # print('Public Metrics', tweet.public_metrics)
            # print("--------")
        # """
        # Response(data=[
        #     <Tweet id=1573347566128865282 
        #     text="Alex Jones testifies in trial over his Sandy Hook hoax lies https://t.co/tDM0ruZaaj via @Yahoo \n\nIt's a good thing for Jones that I wasn't a parent in that courtroom. He is such an arrogant POS.">, <Tweet id=1573347564879245318 text='@INTROVERT__ALEX Bakwaas'>, 
        #     <Tweet id=1573347560672186369 
        #     text='RT @MB_Umek: Sandy Hook Lawyer Attempts To Force Alex Jones To Undergo Communist Struggle Session\n\nhttps://t.co/c921HqehQ8'>, 
        #     <Tweet id=1573347560298782720 
        #     text='There is, as yet, no known defence against the levels of horny this look from @alexxxcoal inflicts.\n\n(The scene is on @Twistys and contains Alex and @MollyStewartXXX being just ridiculously sexy for 35 mins) https://t.co/4RYiQIlG2P'>, 
        #     <Tweet id=1573347559975919616 
        #     text='RT @RonFilipkowski: Heading into court today, Alex Jones claims there is no point to the trial because he’s broke: “You can’t get blood out…'>, 
        #     <Tweet id=1573347559158185984 
        #     text='RT @duty2warn: (1/2) One of the most talked-about subjects on Twitter, rightfully so, is the need for accountability. When we speak to it,…'>, 
        #     <Tweet id=1573347557719379969 
        #     text='Alex Jones is right, they only care when they want to. https://t.co/Fw22KmzaWd'>, <Tweet id=1573347546986160131 text='RT @NicolasGomezMSN: Buenos días, ¿ya renunció Alex Flórez? Ah! ¿Y el petrismo ya se pronunció por el desfalco de los 216 millones de pesos…'>, <Tweet id=1573347545736441856 text='RT @SweeneyABC: An American Airlines passenger punched a flight attendant in the head  after passengers say he was told not to use the firs…'>, <Tweet id=1573347544846983169 text='RT @ArmandoInfo: Las viviendas facturadas por la compañía de Alex Saab en 2014 al Ministerio de Vivienda tuvieron precios de lujo. Un apart…'>], includes={}, errors=[], meta={'newest_id': '1573347566128865282', 'oldest_id': '1573347544846983169', 'result_count': 10, 'next_token': 'b26v89c19zqg8o3fpzbkxhq65xvggo4kb051lfo54pn5p'})
        # """
        # """
        # [<Tweet id=1573348922428461056 
        # text='Alex De Souza İran’lı kadınlara destek olmak için saçlarını kazıttı. https://t.co/zq75raPk4G'>, <Tweet id=1573348920935469056 text='RT @zuricht94: Aparte de Alex Saab, esta sanguijuela @PedroKonductaz es parásito que más ha hecho plata en Venezuela. Todo jalabola busca u…'>, <Tweet id=1573348920377446402 text='@Alex_Salguero88 Que abejorro te ha picado!!!! Vaya bulto te a dejado!!!!'>, <Tweet id=1573348920201457664 text='@Alex_queen12 me gustaria conocerte'>, <Tweet id=1573348917496127488 text='@Hopetraining Should be 20 years. Alex Bellfield got longer for online abuse?'>, <Tweet id=1573348916216864769 text='RT @CanalCotia: O São Paulo fez ótima atuação ontem frente o Penapolense com a expressiva vitória por 6-0. A equipe comandada pelo Alex se…'>, <Tweet id=1573348915776454662 text='RT @SoyRosmy1: El Activista y líder del @troikakollectiv de Miami - Florida se une firme y en Solidaridad Internacional a la lucha por la L…'>, <Tweet id=1573348914555949066 text='RT @MarcialM2: Si @POTUS lo quiere de vuelta, es muy fácil lo que debe hacer:\n\nDerogar las Sanciones Criminales ☠️\nDevolver CITGO y todo lo…'>, <<Tweet id=1573348913846882304 text='@Alex_szn01 Omo 😂😭'>, <Tweet id=1573348912286822400 text='Aşkın olayım - Alex de Souza \nhttps://t.co/lZHXO8I289'>]
        # """
        return pd.DataFrame(data=data, columns=columns)


if __name__ == '__main__':
    twitter_api = TwitterAPI()
    result = twitter_api.search_tweet_by_keyword('Hurricane tracker')
    print(result)
