import pandas as pd


def fscore(predictions, validations):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    dataset = pd.concat([predictions, validations], axis=1)
    dataset.reset_index()
    for i, row in dataset.iterrows():
        if row['predict'] == 'e':
            if row['edible'] == 'e':
                true_pos += 1
            else:
                false_pos += 1
        elif row['predict'] == 'p':
            if row['edible'] == 'p':
                true_neg += 1
            else:
                false_neg += 1
        else:
            pass

    precision = true_pos/(true_pos + false_pos)
    recall = true_pos/(true_pos + false_neg)
    f1score = 2 * (precision * recall)/(precision + recall)

    return precision, recall, f1score
