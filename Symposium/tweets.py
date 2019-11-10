from Symposium.analysis import Analysis
from Symposium.dataloader import load_dataset
from Symposium.nlp import process_text
import pprint

if __name__ == "__main__":
    datafile = "dataset/SentimentAnalysisDataset.csv"
    columns = ["ItemID", "Sentiment", "SentimentSource", "SentimentText"]

    dataset = load_dataset(datafile, columns)

    # dataset analysis
    analysis1 = Analysis(dataset, columns)
    label_splits = analysis1.result_ratio(columns[1])
    mean, median, mode, std, outliers = analysis1.word_count()
    common_words = analysis1.common_words()

    # TODO split test/train

    # summary report printout
    print("*****************************************************************************")
    print("*                                                                           *")
    print("*                     Twitter Sentiment Analysis                            *")
    print("*                                                                           *")
    print("*****************************************************************************")
    print("\nDistribution of label types")
    print("-----------------------------------------------------------------------------")
    print("Negative sentiment:  " + str(label_splits[0]))
    print("Positive sentiment:  " + str(label_splits[1]))
    print("Total :             " + str(label_splits[1] + label_splits[0]))
    print("\n-----------------------------------------------------------------------------")
    print("Tweet word length")
    print("-----------------------------------------------------------------------------")
    print("Mean words:                " + str(mean))
    print("Most frequent word count:  " + str(mode[0][0]) + " words occurred " + str(mode[1][0]) + " times")
    print("Median word count:         " + str(median))
    print("Standard Deviation:        " + str(std))
    print("\n-----------------------------------------------------------------------------")
    print("Most Common Words")
    print("-----------------------------------------------------------------------------")
    pprint.pprint(common_words)
