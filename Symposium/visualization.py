import matplotlib.pyplot as plt
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
    plt.show()
    plt.savefig('graphs/sentiment_distribution.png')


def word_histogram(data):
    sns.set_style('darkgrid')
    sns.distplot(data)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution')
    plt.show()
    plt.savefig('graphs/word_length_distribution.png')


def word_box_plot(data):
    sns.boxplot(x=data)
    plt.show()
# line plot

# write image to file
