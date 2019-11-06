from Symposium.visualization import chart_pie


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

# most common words (graph)



