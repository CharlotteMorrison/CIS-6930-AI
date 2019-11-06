import nltk

def process_text(data, column):
    tokens = [[parse(text) for text in data[column]]]
    return tokens


def parse(df_text):
    token = [word for word in df_text.split()]
    return token


