from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Project2.longevity.graph import plot_grid_search


def run_lr(X_train, X_test, y_train, y_test):

    solver_ncg = ['newton-cg']
    penalty_ncg = ['none', 'l2']
    solver_sag = ['sag']
    penalty_sag = ['none', 'l2']
    solver_saga = ['saga']
    penalty_saga1 = ['none', 'l2', 'l1']
    penalty_saga2 = ['elasticnet']
    solver_lib = ['liblinear']
    penalty_lib = ['l1', 'l2']

    # shared parameters
    iters = [1000, 2000, 3000]
    l1 = [1, 0]

    params = [
        dict(cls__penalty=penalty_ncg, cls__solver=solver_ncg),
        dict(cls__penalty=penalty_sag, cls__solver=solver_sag, cls__max_iter=iters),
        dict(cls__penalty=penalty_saga1, cls__solver=solver_saga, cls__max_iter=iters),
        dict(cls__penalty=penalty_saga2, cls__solver=solver_saga, cls__l1_ratio=l1, cls__max_iter=iters),
        dict(cls__penalty=penalty_lib, cls__solver=solver_lib, cls__max_iter=iters)
    ]

    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA()),
        ('cls', LogisticRegression())
    ])

    grid_search = GridSearchCV(lr_pipe, param_grid=params, cv=10, scoring='f1')
    grid_search.fit(X_train, y_train.values.ravel())

    scores = grid_search.cv_results_

    # plot_grid_search(scores, solvers, penalty_norm, 'Solvers', 'Penalty Norm', 'LinearRegression')

    # test the predictions
    y_predict = grid_search.predict(X_test)
    f1 = f1_score(y_test, y_predict)

    print_report = True
    if print_report:
        report = open("/home/charlotte/PycharmProjects/CIS-6930-AI/Project2/reports/LR.txt", "w+")

        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Logistic Regression Classification")
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
