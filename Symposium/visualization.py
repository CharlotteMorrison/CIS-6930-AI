import matplotlib.pyplot as plt


# pie chart
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
    plt.savefig('graphs/sentiments.png')

    # add save image

# bar graph

# line plot

# write image to file

