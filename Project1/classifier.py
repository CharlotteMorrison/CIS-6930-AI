from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import datetime
import sklearn
from sklearn.tree import DecisionTreeClassifier


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