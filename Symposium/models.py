import time

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

import Symposium.config


def naives_bayes(X_train, y_train, save_loc):
    pipeline = Pipeline([
        ('bow', CountVectorizer(strip_accents='ascii',
                                stop_words='english',
                                lowercase=True)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Na
    ])

    # parameters for grid search
    parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'classifier__alpha': (1e-2, 1e-3),
                  }
    # do 10-fold cross validation for each of the 6 possible combinations of the above params
    grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
    grid.fit(X_train, y_train)
    # summarize results

    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']

    Symposium.config.report.write("\n-----------------------------------------------------------------------------")
    Symposium.config.report.write("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
    Symposium.config.report.write("\n-----------------------------------------------------------------------------\n")
    for mean, stdev, param in zip(means, stds, params):
        Symposium.config.report.write("\nMean: %f Stdev:(%f) with: %r" % (mean, stdev, param))

    # Save the best model
    joblib.dump(grid, save_loc)


def sgd(X_train, y_train, save_loc):
    pipeline_sgd = Pipeline([
        ('bow', CountVectorizer(strip_accents='ascii',
                                stop_words='english',
                                lowercase=True)),
        ('tfidf', TfidfTransformer()),
        ('nb', SGDClassifier()),
    ])
    model = pipeline_sgd.fit(X_train, y_train)

    joblib.dump(model, save_loc)


def svm_linear(X_train, y_train, save_loc):
    # Create feature vectors
    vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    svm_clf = svm.LinearSVC(C=0.1, verbose=1)
    vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
    vec_clf.fit(X_train, y_train)
    joblib.dump(vec_clf, save_loc, compress=3)


def get_predictions(file, X_test, y_test, title):
    # load from file and predict using the best configs found in the CV step
    model = joblib.load(file)
    # get predictions from best model above
    y_preds = model.predict(X_test)

    cm = confusion_matrix(y_test, y_preds)
    Symposium.config.report.write("\n-----------------------------------------------------------------------------\n")
    Symposium.config.report.write(title)
    Symposium.config.report.write("\n-----------------------------------------------------------------------------\n")
    Symposium.config.report.write('accuracy score: ' + str(accuracy_score(y_test, y_preds)))
    Symposium.config.report.write('\n')
    Symposium.config.report.write('confusion matrix: \n')
    Symposium.config.report.write("{} {}\n".format(str(cm[0][0]), str(cm[0][1])))
    Symposium.config.report.write("{} {}\n".format(str(cm[1][0]), str(cm[1][1])))
    Symposium.config.report.write('Classification Report: \n')
    Symposium.config.report.write(classification_report(y_test, y_preds))
