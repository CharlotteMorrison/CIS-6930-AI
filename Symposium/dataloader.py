import pandas as pd


def load_dataset(file, columns):
    dataset = pd.read_csv(file, usecols=columns)
    return dataset
