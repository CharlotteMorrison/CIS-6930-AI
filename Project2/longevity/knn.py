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

    # knn = KNeighborsClassifier()

    # using grid search for k = 2-10, 10 fold cross validation, f1 scoring, run in parallel



