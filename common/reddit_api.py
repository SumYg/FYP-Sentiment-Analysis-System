import praw
# from praw.models import MoreComments
# import pprint
from datetime import datetime
import pandas as pd
import logging

class RedditAPI:
    def __init__(self) -> None:
        with open('./cred/reddit.txt', 'r') as f:
            cred = f.read().split('\n')
        self.reddit = praw.Reddit(
            client_id=cred[0],
            client_secret=cred[1],
            user_agent=cred[2],
        )
        # print(reddit.read_only)

        # search from most of the subreddits
        self.all = self.reddit.subreddit('all')

    class Comments:
        def __init__(self) -> None:
            self.comments = []
            self.comment_columns = ('parent_id', 'comment_id', 'body', 'created_at_utc', 'score', 'ups', 'downs', 'stickied', 'depth')

        def get_all_comments(self, submission):
            # iterate through comments from submission
            # url = "https://www.reddit.com/r/funny/comments/3g1jfi/buttons/"
            # submission = self.reddit.submission(url=url)
            submission.comments.replace_more(limit=0)  # get only first round of comments
            # counter = 0
            submission_id = submission.id
            # for top_level_comment in submission.comments:
            for top_level_comment in submission.comments.list():
                # if isinstance(top_level_comment, MoreComments):
                #     continue
                # print(top_level_comment.body)
                # pprint.pprint(vars(top_level_comment))
                self.comments.append((submission_id, top_level_comment.id, top_level_comment.body, datetime.utcfromtimestamp(top_level_comment.created_utc)
                    , top_level_comment.score, top_level_comment.ups, top_level_comment.downs, top_level_comment.stickied, top_level_comment.depth))
                # counter += 1
                # break
            # print(counter)

    def search_keyword(self, keyword, limit=1000, time_filter='day'):
        """
        Return (Submissions DataFrame, Comments DataFrame)
        """
        data = []
        columns = ('reddit_id', 'title', 'text', 'created_at_utc', 'comment_count', 'score', 'ups', 'downs', 'spoiler', 'stickied', 'upvote_ratio',
                    'is_original_content', 'is_self')

        comments = self.Comments()

        logging.info("Request Reddit API")
        for submission in self.all.search(keyword, limit=limit, time_filter=time_filter):
            # print(type(submission), submission)
            # print(submission.id)
            # print(submission.title)
            # print(submission.selftext)
            # print(submission.created_utc, datetime.utcfromtimestamp(submission.created_utc))
            # print(submission.num_comments)
            # print(submission.score)

            data.append((submission.id, submission.title, submission.selftext, datetime.utcfromtimestamp(submission.created_utc)
                , submission.num_comments, submission.score, submission.ups, submission.downs, submission.spoiler, submission.stickied, submission.upvote_ratio
                ,submission.is_original_content, submission.is_self))
            logging.info("Start to get Comments")
            comments.get_all_comments(submission)
            logging.info("Got all Comments")
            # break
        logging.info("Request Reddit API Finished")
        
        return pd.DataFrame(data=data, columns=columns), pd.DataFrame(data=comments.comments, columns=comments.comment_columns)

if __name__ == '__main__':

    reddit_api = RedditAPI()
    reddit_api.search_keyword('Disney Plus')

    # for submission in reddit.subreddit("learnpython").hot(limit=10):
    #     print(submission.title)
    #     print(submission.score)
    #     print(submission.id)
    #     print(submission.body)
    #     top_level_comments = list(submission.comments)
    #     print(len(top_level_comments), top_level_comments)
    #     all_comments = submission.comments.list()
    #     print(len(all_comments), all_comments)
    #     print("======")
    #     break
    # target = reddit.submission('xhw631')
    # print(target.score)
    # print(target.upvote_ratio)
    # print(target.selftext)
    # print(target.title)
    # print(target.url)
    # print(target.num_comments)  # may not match with the correct number, contains the count for deleted comments, ...

    
    # submission = reddit.submission("39zje0")
    # print(submission.title)  # to make it non-lazy
    # pprint.pprint(vars(submission))

    

