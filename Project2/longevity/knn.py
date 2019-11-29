from sklearn.metrics import f1_score
from Project2.longevity.graph import plot_grid_search
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

# Using continuous data
# Distance measures: Euclidean distance, Manhattan distance

# Data must be normalized (convert range between 0.0 and 1.0) if the scale (range) of the data is not consistent.
# If normalization is not done, features with larger ranges will have outsized influence on the model and features
# with smaller ranges won't have appropriate impact.

# Data reduction justifications:
#
# 1. There are too many missing values (done in main).
# 2. The variance is too low, then the attribute is not meaningful.
# 3. Attributes have too high of a correlation with each other.


def run_knn(X_train, X_test, y_train, y_test):
    # parameter lists
    k_list = list(range(2, 21))  # k values
    d_weight = ['uniform', 'distance']

    # need to update graph extensively for these- and changing them doesn't significantly impact performance
    # scalers = [StandardScaler(), RobustScaler(), QuantileTransformer()]
    params = dict(knn__n_neighbors=k_list, knn__weights=d_weight)

    knn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA()),
        ('knn', KNeighborsClassifier())
    ])

    # using grid search for k = 2-10, 10 fold cross validation, f1 scoring, run in parallel
    grid_search = GridSearchCV(knn_pipe, param_grid=params, cv=10, scoring='f1')
    grid_search.fit(X_train, y_train.values.ravel())

    scores = grid_search.cv_results_

    plot_grid_search(scores, k_list, d_weight, 'N Estimators', 'Max Features', 'KNN')

    # test the predictions
    y_predict = grid_search.predict(X_test)
    f1 = f1_score(y_test, y_predict)

    print_report = True
    if print_report:

        report = open("/home/charlotte/PycharmProjects/CIS-6930-AI/Project2/reports/KNN.txt", "w")

        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("KNN Classification")
        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Mean Train Score:\n {}\n".format(scores['mean_test_score']))
        report.write("Standard Scores:\n{}\n".format(scores['std_test_score']))
        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Best Score: {}\n".format(grid_search.best_score_))
        report.write("Best Parameters: {}\n".format(grid_search.best_params_))
        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Test F1 Score: {}".format(f1))
        report.write("\n-----------------------------------------------------------------------------\n")
        report.close()




