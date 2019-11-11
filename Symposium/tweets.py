import copy
from Symposium.analysis import Analysis, word_cloud
from Symposium.dataloader import load_dataset
from Symposium.models import naives_bayes, get_predictions, svm_linear
from Symposium.preprocessing import *
from sklearn.model_selection import train_test_split
import Symposium.config

if __name__ == "__main__":
    Symposium.config.report.write("*****************************************************************************\n")
    Symposium.config.report.write("*                                                                           *\n")
    Symposium.config.report.write("*                     Twitter Sentiment Analysis                            *\n")
    Symposium.config.report.write("*                                                                           *\n")
    Symposium.config.report.write("*****************************************************************************\n")

    datafile = "dataset/SentimentAnalysisDataset.csv"
    columns = ["ItemID", "Sentiment", "SentimentSource", "SentimentText"]

    dataset = load_dataset(datafile, columns)

    # raw dataset analysis
    raw_analysis = Analysis(dataset, columns)
    label_splits = raw_analysis.result_ratio(columns[1])
    mean, median, mode, std, outliers = raw_analysis.word_count()
    common_words = raw_analysis.common_words()

    Symposium.config.report.write("\nDistribution of label types\n")
    Symposium.config.report.write("\n-----------------------------------------------------------------------------")
    Symposium.config.report.write("\nNegative sentiment:  " + str(label_splits[0]))
    Symposium.config.report.write("\nPositive sentiment:  " + str(label_splits[1]))
    Symposium.config.report.write("\nTotal :             " + str(label_splits[1] + label_splits[0]))
    Symposium.config.report.write("\n-----------------------------------------------------------------------------")
    Symposium.config.report.write("\nTweet word length")
    Symposium.config.report.write("\n-----------------------------------------------------------------------------")
    Symposium.config.report.write("\nMean words:                " + str(mean))
    Symposium.config.report.write("\nMost frequent word count:  " + str(mode[0][0]) + " words occurred "
                                   + str(mode[1][0]) + " times")
    Symposium.config.report.write("\nMedian word count:         " + str(median))
    Symposium.config.report.write("\nStandard Deviation:        " + str(std))
    Symposium.config.report.write("\n-----------------------------------------------------------------------------")
    Symposium.config.report.write("\nMost Common Words")
    Symposium.config.report.write("\n-----------------------------------------------------------------------------\n")
    for word in common_words:
        Symposium.config.report.write(word[0] + "\n")

    # preprocess the dataset manually for analysis
    # filter out the non-text stuff (emojis, hashtags, urls)
    # processed_dataset = copy.deepcopy(dataset)
    # processed_dataset['SentimentText'] = processed_dataset['SentimentText'].apply(process_tweet)
    # processed_dataset['Tokens'] = processed_dataset['SentimentText'].apply(text_process)
    # processed_analysis = Analysis(processed_dataset, columns.append('Tokens'))
    # processed_analysis.common_words(name='preprocess', title='Most common 25 words after text cleaning', column='Tokens')
    # word_cloud(processed_dataset['tokens'])

    # start the training/testing
    # create a pipeline for the data using Naive Bayes Classifier
    X_train, X_test, y_train, y_test = train_test_split(dataset['SentimentText'], dataset['Sentiment'], test_size=0.2)

    # save_NB = "pickles/naive_bayes.pkl"
    # naives_bayes(X_train, y_train, save_NB)
    # get_predictions(save_NB, X_test, y_test, "Naives Bayes Best Model Results")

    save_SVM = "pickles/svm.pkl"
    svm_linear(X_train, y_train, save_SVM)
    get_predictions(save_SVM, X_test, y_test, "Support Vector Machine Model Results")
