import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

import Symposium.config


def naives_bayes(X_train, y_train, save_loc):
    pipeline = Pipeline([
            ('bow', CountVectorizer(strip_accents='ascii',
                                    stop_words='english',
                                    lowercase=True)), # strings to token integer counts
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


def svm_linear(X_train, y_train, save_loc):
    pipeline = Pipeline([
            ('bow', CountVectorizer(strip_accents='ascii',
                                    stop_words='english',
                                    lowercase=True)), # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('classifier', svm.SVC(kernel='linear')),  # train on TF-IDF vectors w/ Na
            ])

    pipeline.fit(X_train, y_train)

    # Save the model
    joblib.dump(pipeline, save_loc)


def get_predictions(file, X_test, y_test, title):
    # load from file and predict using the best configs found in the CV step
    model = joblib.load(file)
    # get predictions from best model above
    y_preds = model.predict(X_test)

    cm = confusion_matrix(y_test, y_preds)
    Symposium.config.report.write("\n-----------------------------------------------------------------------------")
    Symposium.config.report.write(title)
    Symposium.config.report.write("\n-----------------------------------------------------------------------------")
    Symposium.config.report.write('accuracy score: ' + str(accuracy_score(y_test, y_preds)))
    Symposium.config.report.write('\n')
    Symposium.config.report.write('confusion matrix: \n')
    Symposium.config.report.write("{} {}".format(str(cm[0][0]), str(cm[0][1])))
    Symposium.config.report.write("{} {}".format(str(cm[1][0]), str(cm[1][1])))
    Symposium.config.report.write('Classification Report: \n')
    Symposium.config.report.write(classification_report(y_test, y_preds))
