import copy

import numpy as np
import pandas as pd
from Project1.predict import predict
from Project1.id3algorithm import ID3Algorithm


def load_data_samples():
    col_names = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment',
                 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

    features = col_names[1:]
    label = col_names[0]

    data_files = ['training_aa.data', 'training_ab.data', 'training_ac.data', 'training_ad.data', 'training_ae.data',
                  'training_af.data', 'training_ag.data', 'training_ah.data', 'training_ai.data', 'training_aj.data']

    # load data files into pandas dataframe
    list_data_files = [pd.read_csv('datasets/' + data_file, header=None, names=col_names) for data_file in data_files]

    # rename files for easier comprehension
    for dataframe, filename in zip(list_data_files, data_files):
        dataframe['filename'] = filename.replace('.data', '')

    # load the testing data
    test_data = pd.read_csv('datasets/testing.data', header=None, names=col_names)

    return list_data_files, features, label, test_data


if __name__ == "__main__":

    training_data, test_features, label, testing_data = load_data_samples()

    for index in range(len(training_data)):
        test_set = training_data[index]
        if index < len(training_data):
            id3 = ID3Algorithm()
            train_set = training_data[0:index] + training_data[index + 1:]
            train_set = pd.concat(train_set)

            print('=================================')
            print('ID3 Algorithm data set number: ' + str(index + 1))
            print('=================================')

            result = id3.run_id3_algorithm(train_set, label, test_features, train_set)
            id3.graph_it(index)
            predictions = pd.DataFrame(columns=['predict'])

            for i, row in test_set.iterrows():
                predictions.loc[i, 'predict'] = predict(result, row[1:-1])


            totals = predictions['predict'].isin(test_set[label]).value_counts()

            accuracy = totals/len(test_set)
            print("Accuracy for decision tree " + str(index + 1) + ":  " + str(accuracy))

