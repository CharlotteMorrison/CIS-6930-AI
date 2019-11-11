import copy

from Symposium.analysis import Analysis, word_cloud
import Symposium.config
from Symposium.preprocessing import process_tweet, text_process


def data_explorer(dataset, columns):
    Symposium.config.report.write("*****************************************************************************\n")
    Symposium.config.report.write("*                                                                           *\n")
    Symposium.config.report.write("*                     Twitter Sentiment Analysis                            *\n")
    Symposium.config.report.write("*                                                                           *\n")
    Symposium.config.report.write("*****************************************************************************\n")
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

    # preprocess the dataset manually for analysis (this takes FOREVER>>>>)
    # filter out the non-text stuff (emojis, hashtags, urls)
    processed_dataset = copy.deepcopy(dataset)
    processed_dataset['SentimentText'] = processed_dataset['SentimentText'].apply(process_tweet)
    processed_dataset['Tokens'] = processed_dataset['SentimentText'].apply(text_process)
    processed_analysis = Analysis(processed_dataset, columns.append('Tokens'))
    processed_analysis.common_words(name='preprocess', title='Most common 25 words after text cleaning', column='Tokens')
    word_cloud(processed_dataset['tokens'])
