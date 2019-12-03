from dateutil.parser import parse

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from Symposium.dataloader import load_dataset
from Symposium.preprocessing import process_tweet, text_process, stem

if __name__ == "__main__":

    datafile = "dataset/IRAhandle_tweets_1.csv"
    columns = ["external_author_id", "content", "account_type", "publish_date"]

    dataset = load_dataset(datafile, columns)

    # remove the time from date/time, convert to date format
    dataset['publish_date'] = [parse(text.split(' ', 1)[0]) for text in dataset['publish_date']]

    # clean the dataset
    dataset['content'] = dataset['content'].apply(process_tweet)

    # load model(s)
    model_NB = joblib.load("pickles/naive_bayes.pkl")
    # model_SGD = joblib.load("pickles/sgd.pkl")
    # model_SVM = joblib.load("pickles/svm.pkl")

    # produce predictions
    dataset['score'] = model_NB.predict(dataset['content'])

    data_group = dataset.groupby(['publish_date'])['score'].mean()
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    data_group.plot(ax=ax)
    # dataset.groupby(['publish_date']).count()['score'].plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Sentiment')
    plt.title('Troll Tweet Mean Sentiment')
    plt.show()
    fig.savefig('graphs/tweets_1_NB.png')
    plt.close('all')

    # nb = model_NB.predict(tweet)[0]
    # sgd = model_SGD.predict(tweet)[0]
    # svm = model_SVM.predict(tweet)[0]
