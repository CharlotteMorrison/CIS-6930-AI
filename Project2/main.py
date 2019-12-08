from Project2.dataexplorer import explore, correlation_graph
from Project2.longevity.ann import run_ann
from Project2.longevity.knn import run_knn
from Project2.longevity.logisticregression import run_lr
from Project2.longevity.randomforest import run_rf
from Project2.shopping.clan import run_clan
from sklearn.model_selection import train_test_split

from Project2.shopping.kmeans import run_kmeans

if __name__ == "__main__":
    # -----------------------------------------------------------------------------------
    # load and run exploration on the dataset(s)
    # -----------------------------------------------------------------------------------
    nba_dataset = explore("datasets/nba.csv", "NBA")
    shop_dataset = explore("datasets/online_shoppers_intention.csv", "Online_Shoppers")

    # -----------------------------------------------------------------------------------
    # remove empty rows (the exploration showed no empty values in online, only nba
    # -----------------------------------------------------------------------------------
    nba_dataset.dropna(axis='index', inplace=True)

    # -----------------------------------------------------------------------------------
    # split the data set into attributes and target values (omit name)
    # -----------------------------------------------------------------------------------
    nba_attributes = nba_dataset[nba_dataset.columns[1:-1]]
    nba_target = nba_dataset[nba_dataset.columns[-1:]]

    shop_attributes = shop_dataset[shop_dataset.columns[0:-1]]
    shop_target = shop_dataset[shop_dataset.columns[-1:]]

    # -----------------------------------------------------------------------------------
    # split the dataset into test/train
    # -----------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(nba_attributes, nba_target, test_size=0.2)

    # -----------------------------------------------------------------------------------
    # check attribute correlations
    # -----------------------------------------------------------------------------------
    correlation_graph(nba_attributes, "NBA")
    correlation_graph(shop_attributes, "Online_shopping")

    # -----------------------------------------------------------------------------------
    # Run analysis
    # -----------------------------------------------------------------------------------
    run_knn(X_train, X_test, y_train, y_test)
    run_rf(X_train, X_test, y_train, y_test)
    run_lr(X_train, X_test, y_train, y_test)
    run_ann(X_train, X_test, y_train, y_test)
    run_kmeans(shop_attributes, shop_target, "K-Means")
    run_clan(shop_attributes, shop_target)


