from Symposium.analysis import Analysis
from Symposium.dataloader import load_dataset
from Symposium.nlp import process_text

if __name__ == "__main__":
    datafile = "dataset/SentimentAnalysisDataset.csv"
    columns = ["ItemID", "Sentiment", "SentimentSource", "SentimentText"]

    # split test/train

    dataset = load_dataset(datafile, columns)

    # dataset analysis
    analysis1 = Analysis(dataset, columns)
    analysis1.result_ratio(columns[1])



