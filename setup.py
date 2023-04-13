import sys

def load():
    paths = ['common/','SentenceVAE/']
    for p in paths:
        sys.path.insert(0, p)