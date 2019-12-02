from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


@ignore_warnings(category=ConvergenceWarning)
def run_kmeans(X, y, name):
    report = open("/home/charlotte/PycharmProjects/CIS-6930-AI/Project2/reports/k_means.txt", "w")
    report.write("\n-----------------------------------------------------------------------------\n")
    report.write("K Means Classification")
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
    # n_clusters is the number of clusters
    # init is a random point
    # n_init will run the algorithm 10 times and choose the best
    # max_iter will prevent it from running indefinitely
    # tol is the tolerance to changes in the within cluster sum squared error.
    km = KMeans(
        n_clusters=4, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )

    y_predict = km.fit_predict(X_scaled)
    total_rand_score = adjusted_rand_score(y.values.ravel(), y_predict)
    report.write("\n-----------------------------------------------------------------------------\n")
    report.write('Total Rand score (all attributes: {})\n'.format(total_rand_score))
    report.write('Cluster Center: \n{}\n'.format(km.cluster_centers_))
    report.write('Number of iterations: {}\n'.format(km.n_iter_))
    report.write("\n-----------------------------------------------------------------------------\n")

    c1_count = 0
    for C1 in X_scaled.T:

        temp_array = np.expand_dims(C1, axis=1)
        c2_count = 1
        for C2 in X_scaled.T:
            # compares all two pairs of attributes
            X_temp = np.vstack((C1, C2)).T
            y_predict = km.fit_predict(X_temp)
            rand_score = adjusted_rand_score(y.values.ravel(), y_predict)

            if rand_score >= 0.25:
                # plot the 4 clusters
                graph_it(km, X_temp, y_predict, 'Pair_{}_{}\n'.format(labels[c1_count], labels[c2_count - 1]))
                report.write("\n-----------------------------------------------------------------------------\n")
                report.write('Pair: {}, {}\n'.format(labels[c1_count], labels[c2_count - 1]))
                report.write('Cluster Center: \n{}\n'.format(km.cluster_centers_))
                report.write('Number of iterations: {}\n'.format(km.n_iter_))
                report.write('Rand Score: {}\n'.format(rand_score))
                report.write("\n-----------------------------------------------------------------------------\n")

            C2_temp = np.expand_dims(C2, axis=1)
            temp_array = np.concatenate((temp_array, C2_temp), axis=1)
            y_predict = km.fit_predict(temp_array)
            rand_score = adjusted_rand_score(y.values.ravel(), y_predict)

            if rand_score >= 0.25:
                report.write("\n-----------------------------------------------------------------------------\n")
                report.write('Pair: {}, {}\n'.format(labels[c1_count], labels[0:c2_count]))
                report.write('Cluster Center: \n{}\n'.format(km.cluster_centers_))
                report.write('Number of iterations: {}\n'.format(km.n_iter_))
                report.write('Rand Score: {}\n'.format(rand_score))
                report.write("\n-----------------------------------------------------------------------------\n")

            c2_count += 1
        c1_count += 1
    report.close()


def graph_it(km, X_temp, y_predict, title):
    plt.scatter(
        X_temp[y_predict == 0, 0], X_temp[y_predict == 0, 1],
        s=50, c='lightgreen',
        marker='s', edgecolor='black',
        label='cluster 1'
    )

    plt.scatter(
        X_temp[y_predict == 1, 0], X_temp[y_predict == 1, 1],
        s=50, c='orange',
        marker='o', edgecolor='black',
        label='cluster 2'
    )

    plt.scatter(
        X_temp[y_predict == 2, 0], X_temp[y_predict == 2, 1],
        s=50, c='lightblue',
        marker='v', edgecolor='black',
        label='cluster 3'
    )
    plt.scatter(
        X_temp[y_predict == 3, 0], X_temp[y_predict == 3, 1],
        s=50, c='yellow',
        marker='D', edgecolor='black',
        label='cluster 4'
    )
    # plot the centroids
    plt.scatter(
        km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
        s=250, marker='X',
        c='red', edgecolor='black',
        label='centroids'
    )
    plt.title(title)
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.savefig('/home/charlotte/PycharmProjects/CIS-6930-AI/Project2/plots/{}.png'.format(title))
    plt.show()
