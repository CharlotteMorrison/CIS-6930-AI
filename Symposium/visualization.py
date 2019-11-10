import matplotlib.pyplot as plt
import nltk
import seaborn as sns

# pie chart
import numpy as np


def chart_pie(sizes, labels):
    # Data to plot
    labels = labels
    sizes = sizes
    colors = ['lightcoral', 'lightskyblue']

    # Plot
    plt.title("Tweet Sentiment Distribution")
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')

    plt.savefig('graphs/sentiment_distribution.png')
    plt.show()


def word_histogram(data):
    sns.set_style('darkgrid')
    sns.distplot(data)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution')
    plt.savefig('graphs/word_length_distribution.png')
    plt.show()


def word_histogram_outcome(pos, neg):
    plt.figure(figsize=(12, 6))
    plt.xlim(0, 45)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution by Sentiment')
    plt.hist([pos, neg], color=['r', 'b'], alpha=0.5, label=['positive', 'negative'])
    plt.legend(loc='upper right')
    plt.savefig('graphs/word_length_distribution_by_sentiment.png')


def word_box_plot(data):
    sns.boxplot(x=data)
    plt.xlabel('Word Count')
    plt.title('Word Count Box Plot')
    plt.savefig('graphs/word_length_box_plot.png')
    plt.show()


# line plot
def plot_common_words(words):
    plt.figure(figsize=(12, 5))
    plt.title('Top 25 most common words')
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(words)
    fd.plot(25, cumulative=False)
    plt.savefig('graphs/common_words_count.png')
    plt.show()
# write image to file
