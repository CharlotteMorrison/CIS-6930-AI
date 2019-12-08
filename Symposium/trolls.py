from datetime import date
from dateutil.parser import parse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from Symposium.dataloader import load_dataset
from Symposium.preprocessing import process_tweet, text_process, stem

if __name__ == "__main__":
    # load model(s)
    # model_NB = joblib.load("pickles/naive_bayes.pkl")
    # model_SGD = joblib.load("pickles/sgd.pkl")
    model_SVM = joblib.load("pickles/svm.pkl")

    dataset_graph = pd.Series()
    for x in range(1, 13):
        datafile = "dataset/IRAhandle_tweets_{}.csv".format(x)
        columns = ["content", "publish_date", "language"]

        dataset = load_dataset(datafile, columns)

        # the model is trained on english, so only keep the English tweets
        dataset = dataset[dataset.language.str.contains('English')]

        # remove the time from date/time, convert to date format
        dataset['publish_date'] = [parse(text.split(' ', 1)[0]) for text in dataset['publish_date']]

        # limit to tweets after 12/1/2015
        dataset = dataset[dataset['publish_date'] > pd.Timestamp(date(2015, 1, 1))]

        # clean the dataset
        dataset['content'] = dataset['content'].apply(str)
        dataset['content'] = dataset['content'].apply(process_tweet)

        # produce predictions
        dataset['score'] = model_SVM.predict(dataset['content'])

        dataset_graph = pd.concat([dataset_graph, dataset], axis=0, sort=True)

        print("Processed dataset {}".format(x))

    data_mean = dataset_graph.groupby(['publish_date'])['score'].mean()
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 7))
    data_mean.plot(ax=ax)
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Sentiment')
    plt.title('Troll Tweet Mean Sentiment')
    plt.show()
    fig.savefig('graphs/tweets_1_NB.png')
    plt.close('all')
