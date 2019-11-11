import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from wordcloud import WordCloud

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
    plt.close('all')


def word_histogram(data):
    sns.set_style('darkgrid')
    sns.distplot(data)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution')
    plt.savefig('graphs/word_length_distribution.png')
    plt.show()
    plt.close('all')


def word_box_plot(data):
    sns.boxplot(x=data)
    plt.xlabel('Word Count')
    plt.title('Word Count Box Plot')
    plt.savefig('graphs/word_length_box_plot.png')
    plt.show()
    plt.close('all')


# line plot
def plot_common_words(words, name, title):
    fig = plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.xticks(fontsize=13, rotation=90)
    fd = nltk.FreqDist(words)
    fd.plot(25, cumulative=False)
    plt.show()
    fig.savefig('graphs/common_words_count_' + name + '.png')
    plt.close('all')


def word_histogram_outcome(pos, neg):
    plt.figure(figsize=(12, 6))
    plt.xlim(0, 45)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution by Sentiment')
    plt.hist([pos, neg], color=['r', 'b'], alpha=0.5, label=['positive', 'negative'])
    plt.legend(loc='upper right')
    plt.savefig('graphs/word_length_distribution_by_sentiment.png')
    plt.show()
    plt.close('all')


def word_cloud(words):
    all_words = []
    for line in list(words):
        words = line.split()
        for word in words:
            all_words.append(word.lower())
    # for line in words:
        # all_words.extend(line)
    # creates a word frequency dictionary
    word_freq = nltk.Counter(all_words)
    # draw a Word Cloud with word frequencies
    wordcloud = WordCloud(width=900,
                          height=500,
                          max_words=500,
                          max_font_size=100,
                          relative_scaling=0.5,
                          colormap='Blues',
                          normalize_plurals=True).generate_from_frequencies(word_freq)
    plt.figure(figsize=(17, 14))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('graphs/frequent_word_cloud.png')
    plt.show()
    plt.close('all')
