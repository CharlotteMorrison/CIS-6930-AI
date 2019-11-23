from Project2.dataexplorer import explore
from Project2.longevity.knn import run_knn


if __name__ == "__main__":

    # load and run exploration on the dataset(s)
    # -----------------------------------------------------------------------------------
    nba_dataset = explore("datasets/nba.csv", "NBA")
    online_dataset = explore("datasets/online_shoppers_intention.csv", "Online_Shoppers")

    # remove empty rows (the exploration showed no empty values in online, only nba
    # -----------------------------------------------------------------------------------
    nba_dataset.dropna(axis='index', inplace=True)

    # split the data set into attributes and target values (omit name)
    attributes = nba_dataset[nba_dataset.columns[1:-1]]
    target = nba_dataset[nba_dataset.columns[-1:]]

    # Run K-Nearest Neighbors analysis
    run_knn(attributes, target)





