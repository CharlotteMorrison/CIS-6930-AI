from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score


@ignore_warnings(category=ConvergenceWarning)
@ignore_warnings(category=DataConversionWarning)
def run_clan(X, y):
    report = open("/home/charlotte/PycharmProjects/CIS-6930-AI/Project2/reports/agglomerative.txt", "w")
    report.write("\n-----------------------------------------------------------------------------\n")
    report.write("Complete-Linkage Agglomerative Classification")
    report.write("\n-----------------------------------------------------------------------------\n")

    labels = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
              'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
              'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',
              'Weekend', 'Revenue']

    # transform the categorical data using one hot encoding
    # get categorical columns and convert to integer values
    X_cat = X[X.columns[-8:]]
    le = preprocessing.LabelEncoder()
    X_num = X_cat.apply(le.fit_transform)

    # use integer values to one-hot encode into a sparse matrix
    # one-hot does not perform well
    # encoder = preprocessing.OneHotEncoder()
    # encoder = preprocessing.OneHotEncoder()
    # encoder.fit(X_num)
    # X_hot = encoder.transform(X_num).toarray()

    X_encoded = np.concatenate((X[X.columns[:-8]], X_num), axis=1)

    # scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    ac = AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
                                 connectivity=None, distance_threshold=None,
                                 linkage='ward', memory=None, n_clusters=4,
                                 pooling_func='deprecated')

    y_predict = ac.fit_predict(X_scaled)
    total_rand_score = adjusted_rand_score(y.values.ravel(), y_predict)
    dbi_score = davies_bouldin_score(y.values.ravel().reshape(-1, 1), y_predict.reshape(-1, 1))

    report.write("\n-----------------------------------------------------------------------------\n")
    report.write('Total Rand score (all attributes: {})\n'.format(total_rand_score))
    report.write('Number of Leaves: {}\nNumber of Children: {}\n'.format(ac.n_leaves_, ac.children_))
    report.write('Number of Connected Components: {}\n'.format(ac.n_connected_components_))
    report.write('Davies-Bouldin Index: {}\n'.format(dbi_score))
    report.write("\n-----------------------------------------------------------------------------\n")

    c1_count = 0
    for C1 in X_scaled.T:

        temp_array = np.expand_dims(C1, axis=1)
        c2_count = 1
        for C2 in X_scaled.T:
            # compares all two pairs of attributes
            X_temp = np.vstack((C1, C2)).T
            y_predict = ac.fit_predict(X_temp)
            rand_score = adjusted_rand_score(y.values.ravel(), y_predict)
            dbi_score = davies_bouldin_score(y.values.ravel().reshape(-1, 1), y_predict.reshape(-1, 1))

            if rand_score >= 0.25 or dbi_score >= 200.25:
                # plot the 4 clusters
                report.write("\n-----------------------------------------------------------------------------\n")
                report.write('Pair: {}, {}\n'.format(labels[c1_count], labels[c2_count - 1]))
                report.write('Total Rand score (all attributes: {})\n'.format(total_rand_score))
                report.write('Number of Leaves: {}\nNumber of Children: {}\n'.format(ac.n_leaves_, ac.children_))
                report.write('Number of iterations: {}\n'.format(ac.n_connected_components_))
                report.write('Davies-Bouldin Index: {}\n'.format(dbi_score))
                report.write("\n-----------------------------------------------------------------------------\n")

            C2_temp = np.expand_dims(C2, axis=1)
            temp_array = np.concatenate((temp_array, C2_temp), axis=1)
            y_predict = ac.fit_predict(temp_array)
            rand_score = adjusted_rand_score(y.values.ravel(), y_predict)
            dbi_score = davies_bouldin_score(y.values.ravel().reshape(-1, 1), y_predict.reshape(-1, 1))

            if rand_score >= 0.25 or dbi_score >= 200.25:
                report.write("\n-----------------------------------------------------------------------------\n")
                report.write('Pair: {}, {}\n'.format(labels[c1_count], labels[0:c2_count]))
                report.write('Total Rand score (all attributes: {})\n'.format(total_rand_score))
                report.write('Number of Leaves: {}\nNumber of Children: {}\n'.format(ac.n_leaves_, ac.children_))
                report.write('Number of iterations: {}\n'.format(ac.n_connected_components_))
                report.write('Davies-Bouldin Index: {}\n'.format(dbi_score))
                report.write("\n-----------------------------------------------------------------------------\n")

            c2_count += 1
        c1_count += 1
    report.close()
