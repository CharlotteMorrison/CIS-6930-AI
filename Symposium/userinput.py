import PySimpleGUI as sg
import joblib


# formatting helper
def sentiment_str(x):
    if x == 0:
        return 'Negative'
    else:
        return 'Positive'


def analyze(tweet):
    nb = sentiment_str(model_NB.predict(tweet)[0])
    sgd = sentiment_str(model_SGD.predict(tweet)[0])
    svm = sentiment_str(model_SVM.predict(tweet)[0])
    # add in a method for disagreement
    return nb, sgd, svm


if __name__ == "__main__":
    # load model(s)
    model_NB = joblib.load("pickles/naive_bayes.pkl")
    model_SGD = joblib.load("pickles/sgd.pkl")
    model_SVM = joblib.load("pickles/svm.pkl")

    layout = [[sg.Text('Twitter Sentiment Analysis', size=(75, 1), font=('Helvetica', 24), justification='center')],
              [sg.Text('Enter a possible tweet (280 characters or less):', font=('Helvetica', 16)),
               sg.Input(key='_IN_', do_not_clear=False, size=(75, 5), font=('Helvetica', 16))],
              [sg.Text(size=(80, 1)), sg.Button('Analyze', font=('Helvetica', 16))],
              [sg.Text('Tweet sentiment according to classification models:', font=('Helvetica', 16))],
              [sg.Text(font=('Helvetica', 16),  size=(75, 2), text_color='blue', key='_TWEET_OUT_')],
              [sg.Text('Naives-Bayes                      ', size=(33, 1), font=('Helvetica', 16)),
               sg.Text('Stochastic Gradient Descent       ', size=(33, 1),  font=('Helvetica', 16)),
               sg.Text('Support Vector                    ', size=(33, 1), font=('Helvetica', 16))],
              [sg.Text(size=(33, 2), font=('Helvetica', 16), key='_NB_Output_'),
               sg.Text(size=(33, 2), font=('Helvetica', 16), key='_SGD_Output_'),
               sg.Text(size=(33, 2), font=('Helvetica', 16), key='_SVM_Output_')]]

    window = sg.Window('Tweet Sentiment Analysis', layout)
    while True:  # Event Loop
        event, values = window.read()  # can also be written as event, values = window()
        if event is None or event == 'Exit':
            break
        if event == 'Analyze':
            tweet_in = [values['_IN_']]

            p1, p2, p3 = analyze(tweet_in)

            tweet_out = result = "Your Tweet: '{}'".format(str(values['_IN_']))
            window['_TWEET_OUT_'].update(tweet_out)
            window['_NB_Output_'].update(p1)
            window['_SGD_Output_'].update(p2)
            window['_SVM_Output_'].update(p3)
    window.close()
