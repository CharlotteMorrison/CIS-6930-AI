import pandas as pd
import time


def explore(location, name):
    dataset = pd.read_csv(location)
    dataset.head()
    shape = dataset.shape
    index = dataset.index
    columns = dataset.columns
    missing_values = dataset.isnull().sum()
    print(shape)
    print(index)
    print(columns)
    print(missing_values)

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
