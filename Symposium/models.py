import time

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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


def svm_linear(X_train, y_train, save_loc):
    # Create feature vectors

    vectorizer = TfidfVectorizer(strip_accents='ascii',
                                 stop_words='english',
                                 lowercase=True,
                                 min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(X_train)
    print('complete: train_vectorizer')
    test_vectors = vectorizer.transform(X_train)
    print('complete: test_vectorizer')
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear', verbose=True)
    print('complete: set svm')
    t0 = time.time()
    print(time.asctime(time.localtime(time.time())))
    classifier_linear.fit(train_vectors, y_train)
    print('complete: linear fit')
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    print('complete: predictions')
    t2 = time.time()
    time_linear_train = t1 - t0
    time_linear_predict = t2 - t1
    # results
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))

    joblib.dump(classifier_linear, save_loc)


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
