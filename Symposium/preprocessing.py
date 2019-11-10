# modified from Ronald Wahome
# https://towardsdatascience.com/the-real-world-as-seen-on-twitter-sentiment-analysis-part-one-5ac2d06b63fb
# helper function to clean tweets
import re
import string
from string import punctuation

from nltk.corpus import stopwords


def process_tweet(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+', '', tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase
    tweet = tweet.lower()
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Remove single space remaining at the front of the tweet.
    tweet = tweet.lstrip(' ')
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF')
    return tweet


def text_process(raw_text):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # check for and remove punctuation characters
    no_punc = [char for char in list(raw_text) if char not in string.punctuation]

    # rejoin the characters without the punctuation
    no_punc = ''.join(no_punc)

    # remove all the stop words
    return [word for word in no_punc.lower().split() if word.lower() not in stopwords.words('english')]


# function to remove a custom set of word (not used initially)
def remove_words(word_list):
    # add custom words to the list
    remove = []
    return [word for word in word_list if word not in remove]
