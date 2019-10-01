import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import datetime


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


def run_dt_classifier(X_train_dt, X_test_dt, y_train_dt, y_test_dt, features):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train_dt, y_train_dt)
    y_predict_dt = clf.predict(X_test_dt)
    print("Accuracy: ", metrics.accuracy_score(y_test_dt, y_predict_dt))

    dot_data = sklearn.tree.export_graphviz(clf,  filled=True, rounded=True, special_characters=True,
                                            feature_names=features, class_names=['edible', 'non-edb.'])
    graph = pydotplus.graph_from_dot_data(dot_data)
    date_time = datetime.datetime.now()
    graph.write_png("images/dtclassifier" + date_time.strftime("%Y-%m-%d %H:%M:%S" + ".png"))
    Image(graph.create_png())


if __name__ == "__main__":

    training_data, test_features, label, testing_data = load_data_samples()

    o_encoder = OrdinalEncoder()
    for index in range(len(training_data)):
        test_set = training_data[index]
        if index < len(training_data):
            train_set = training_data[0:index] + training_data[index + 1:]
            train_set = pd.concat(train_set)

            X_train = o_encoder.fit_transform(train_set[test_features])
            X_test = o_encoder.fit_transform(test_set[test_features])
            y_train = o_encoder.fit_transform(train_set[label].values.reshape(-1, 1))
            y_test = o_encoder.fit_transform(test_set[label].values.reshape(-1, 1))

            run_dt_classifier(X_train, X_test, y_train, y_test, test_features)
