from pytrends.request import TrendReq
# from pprint import pprint
import logging
# pytrend = TrendReq(hl='en-US', tz=360)
# keywords = ['intel', 'amd', 'samsung', 'apple', 'tesla', '4']
# pytrend.build_payload(
#      kw_list=keywords,
#      cat=0,
#      timeframe='today 1-m',
#      geo='HK',
#      gprop='')

# pprint(pytrend.interest_over_time())

# pytrends = TrendReq(hl='en-US', tz=360)

# pprint(pytrends.trending_searches(pn='hong_kong'))
# print("=================")

class GoogleTrends():
    def __init__(self) -> None:
        self.pytrend = TrendReq(hl='en-US', tz=360)

    def get_trending_searches(self, pn='united_states'):
        """
        Return the dataframe which contains 20 trending searches.
        A bit different with the site
        https://trends.google.com/trends/trendingsearches/daily?geo=US
        """
        # pytrend = TrendReq(hl='en-US', tz=360)
        logging.info("Get Trending Searches")
        trending_searches = self.pytrend.trending_searches(pn=pn)
        # logging.info(trending_searches)
        logging.info(trending_searches)
        # pprint(pytrend.realtime_trending_searches(pn='US'))
        return trending_searches


    def get_suggestions(self, keyword):
        """
        Get related keywords
        """
        # pytrend = TrendReq(hl='en-US', tz=360)
        logging.info("Get suggestions")
        suggestions = self.pytrend.suggestions(keyword)
        logging.info(suggestions)
        return suggestions

if __name__ == '__main__':
    google_trends = GoogleTrends()
    # google_trends.get_suggestions("Alex")
    print(google_trends.get_trending_searches())
