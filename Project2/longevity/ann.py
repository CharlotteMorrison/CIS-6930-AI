from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Project2.longevity.graph import plot_grid_search


def run_ann(X_train, X_test, y_train, y_test):

    act_list = ['identity', 'logistic', 'tanh', 'relu']
    solver_list = ['lbfgs', 'sgd', 'adam']

    params = dict(ann__activation=act_list, ann__solver=solver_list, ann__max_iter=[500])

    ann_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reduce_dim', PCA()),
        ('ann', MLPClassifier())
    ])

    grid_search = GridSearchCV(ann_pipe, param_grid=params, cv=10, scoring='f1')
    grid_search.fit(X_train, y_train.values.ravel())

    scores = grid_search.cv_results_

    plot_grid_search(scores, act_list, solver_list, 'Activations', 'Solvers', 'ANN')

    # test the predictions
    y_predict = grid_search.predict(X_test)
    f1 = f1_score(y_test, y_predict)

    print_report = True
    if print_report:
        report = open("/home/charlotte/PycharmProjects/CIS-6930-AI/Project2/reports/ANN.txt", "w+")

        report.write("\n-----------------------------------------------------------------------------\n")
        report.write("Artificial Neural Network Classification")
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
