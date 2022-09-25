import praw
with open('./cred/reddit.txt', 'r') as f:
    cred = f.read().split('\n')

reddit = praw.Reddit(
    client_id=cred[0],
    client_secret=cred[1],
    user_agent="my app",
)

print(reddit.read_only)



if __name__ == '__main__':

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
    target = reddit.submission('xhw631')
    print(target.score)
    print(target.upvote_ratio)
    print(target.url)
    print(target.num_comments)
