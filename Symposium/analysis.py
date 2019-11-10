from collections import Counter

from scipy import stats
from Symposium.visualization import chart_pie, word_histogram, word_box_plot, word_histogram_outcome
import numpy as np


class Analysis(object):

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    # number of positive or negative responses (pie chart)
    def result_ratio(self, result_label):
        results = self.dataset[result_label].value_counts()
        label = 'negative', 'positive'
        chart_pie([results[0], results[1]], label)
        return results

    # word count (mean median mode, sd) (bar charts) divide by pos/neg
    def word_count(self):
        def counter(sentence):
            return len(sentence.split())

        self.dataset['WordCount'] = self.dataset[self.labels[3]].apply(counter)
        mean = np.mean(self.dataset['WordCount'], axis=0)
        median = np.median(self.dataset['WordCount'], axis=0)
        mode = stats.mode(self.dataset['WordCount'], axis=0)
        std = np.std(self.dataset['WordCount'], axis=0)

        # graphs
        word_box_plot(self.dataset['WordCount'])
        word_histogram(self.dataset['WordCount'])

        outlier_index = np.where(self.dataset['WordCount'] > 38)[0]
        outliers = [self.dataset.iloc[i] for i in outlier_index]

        pos_data = self.dataset['WordCount'][self.dataset.Sentiment == 1]
        neg_data = self.dataset['WordCount'][self.dataset.Sentiment == 0]

        word_histogram_outcome(pos_data, neg_data)

        return mean, median, mode, std, outliers

    # most common words (graph)
    def common_words(self):
        all_words = []
        for line in list(self.dataset['SentimentText']):
            words = line.split()
            for word in words:
                all_words.append(word.lower())

        return Counter(all_words).most_common(10)
