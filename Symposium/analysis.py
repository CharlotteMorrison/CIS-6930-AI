from scipy import stats
from Symposium.visualization import chart_pie, word_histogram, word_box_plot
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
        word_histogram(self.dataset['WordCount'])
        word_box_plot(self.dataset['WordCount'])
        return mean, median, mode, std


# most common words (graph)



