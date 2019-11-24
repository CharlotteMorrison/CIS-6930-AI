import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt


def explore(location, name):
    dataset = pd.read_csv(location)
    dataset.head()
    shape = dataset.shape
    index = dataset.index
    columns = dataset.columns
    missing_values = dataset.isnull().sum()

    # change flag to update report
    print_report = False
    if print_report:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        report = open("reports/explore_{}_{}.txt".format(name, timestr), "w")

        report.write("\n-----------------------------------------------------------------------------\n")
        report.write(name)
        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Dataset Shape:\n {}\n".format(shape))
        report.write("Dataset Indices:\n {}\n".format(index))
        report.write("Dataset: columns:\n {}\n".format(columns))
        report.write("Dataset Missing Values:\n {}".format(missing_values))
        report.write("\n-----------------------------------------------------------------------------\n")
    return dataset


def correlation_graph(dataset, name):
    corr = dataset.corr().abs()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(dataset.columns), 1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(dataset.columns)
    ax.set_yticklabels(dataset.columns)
    plt.savefig('plots/correlations.png')
    plt.show()
    plt.close('all')

    corr_list = corr.unstack()
    sort_corr_list = corr_list.sort_values(kind='quicksort')
    pd.set_option('display.max_rows', None)

    print_report = True
    if print_report:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        report = open("reports/correlation_{}_{}.txt".format(name, timestr), "w")

        report.write("\n-----------------------------------------------------------------------------\n")
        report.write(name + " Correlation Scores")
        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Correlations: \n{}\n".format(sort_corr_list))
        report.write("\n-----------------------------------------------------------------------------\n")


