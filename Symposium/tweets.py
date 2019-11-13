import joblib
from Symposium.preprocessing import stem, process_tweet
from Symposium.dataexplorer import data_explorer
from Symposium.dataloader import load_dataset
from Symposium.models import naives_bayes, get_predictions, svm_linear
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    datafile = "dataset/SentimentAnalysisDataset.csv"
    columns = ["ItemID", "Sentiment", "SentimentSource", "SentimentText"]

    dataset = load_dataset(datafile, columns)

    # Explore the raw dataset (create graphs)
    # data_explorer(dataset, columns)

    # start the training/testing

    X_train, X_test, y_train, y_test = train_test_split(dataset['SentimentText'], dataset['Sentiment'], test_size=0.2)

    # Uncomment to run Naive Bayes (time consuming)
    # save_NB = "pickles/naive_bayes.pkl"
    # naives_bayes(X_train, y_train, save_NB)
    # get_predictions(save_NB, X_test, y_test, "Naives Bayes Best Model Results")

    # try sgd classifier-

    # Uncomment to run the Support Vector Machine

    # preprocess tweets
    X_train = [stem(process_tweet(tweet)) for tweet in X_train]
    X_test = [stem(process_tweet(tweet)) for tweet in X_test]

    joblib.dump(X_train, "pickles/X_train.pkl")
    joblib.dump(X_test, "pickles/X_test.pkl")

    save_SVM = "pickles/svm.pkl"
    # svm_linear(X_train, y_train, save_SVM)
    get_predictions(save_SVM, X_test, y_test, "Support Vector Machine Model Results")
