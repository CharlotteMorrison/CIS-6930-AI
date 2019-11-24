import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from Project2.longevity.graph import plot_grid_search


def run_rf(X, y):
    # normalization is not needed for random forest.
    # RandomForestClassifier parameters
    # n_estimators (default=10)
    # criterion (gini, entropy)
    # max_features (default=auto)
    # max_depth (default=none), might adjust if overfitting
    # min_samples_split (default=2)
    # min_samples_leaf (default=1)
    # min_weight_fraction_leaf (default=0)
    # min_impurity_decrease (default=0)
    # bootstrap (default=True)  DO NOT CHANGE
    # oob_score (default=False) train on n-1 samples of the data, faster
    # n_jobs (default=1) use -1 to use all available processors
    # random_state (default=0) for replication
    # warm_start (default=false) fits a new forest each time
    # class_weight (defaults weights to one)

    crit = ['gini', 'entropy']
    estimators = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    params = dict(cls__criterion=crit, cls__n_estimators=estimators)

    rf_pipe = Pipeline([
        ('cls', RandomForestClassifier())
    ])

    grid_search = GridSearchCV(rf_pipe, param_grid=params, cv=10, scoring='f1')
    grid_search.fit(X, y.values.ravel())
    scores = grid_search.cv_results_

    plot_grid_search(scores, estimators, crit, 'N Estimators', 'Max Features', 'RandomForest')

    print_report = True
    if print_report:
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        report = open("/home/charlotte/PycharmProjects/CIS-6930-AI/Project2/reports/RF_{}.txt".format(timestr), "w")

        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Random Forest Classification")
        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Mean Train Score:\n {}\n".format(scores['mean_test_score']))
        report.write("Standard Scores:\n{}\n".format(scores['std_test_score']))
        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Best Score: {}\n".format(grid_search.best_score_))
        report.write("Best Parameters: {}\n".format(grid_search.best_params_))
        report.write("\n-----------------------------------------------------------------------------\n")





