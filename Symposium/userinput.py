import PySimpleGUI as sg
import joblib


# formatting helper
def sentiment_str(x):
    if x == 0:
        return 'Negative'
    else:
        return 'Positive'


if __name__ == "__main__":
    # load model(s)
    model_NB = joblib.load("pickles/naive_bayes.pkl")
    model_SGD = joblib.load('pickles/sgd.pkl')
    model_SVM = joblib.load('pickles/svm.pkl')

    layout = [[sg.Text('Twitter Sentiment Analysis', size=(75, 1), font=('Helvetica', 24), justification='center')],
              [sg.Text('Enter a possible tweet (280 characters or less):', font=('Helvetica', 16)),
               sg.Input(key='_IN_', do_not_clear=False, size=(75, 5), font=('Helvetica', 16))],
              [sg.Text('', size=(80, 1)), sg.Button('Analyze', font=('Helvetica', 16))],
              [sg.Text('Tweet sentiment according to classification models:', font=('Helvetica', 16))],
              [sg.Text('Your tweet: ', font=('Helvetica', 16)),
               sg.Text('', font=('Helvetica', 16), text_color='blue', key='_TWEET_OUT_')],
              [sg.Text("Naives-Bayes", font=('Helvetica', 16)),
               sg.Text("Stochastic Gradient Descent", font=('Helvetica', 16)),
               sg.Text("Support Vector", font=('Helvetica', 16))],
              [sg.Text(size=(33, 5), font=('Helvetica', 16), key='_NB_Output_'),
               sg.Text(size=(33, 5), font=('Helvetica', 16), key='_SGD_Output_'),
               sg.Text(size=(33, 5), font=('Helvetica', 16), key='_SVM_Output_')]]

    window = sg.Window('Tweet Sentiment Analysis', layout)

    while True:  # Event Loop
        event, values = window.read()  # can also be written as event, values = window()
        print(event, values)
        if event is None or event == 'Exit':
            break
        if event == 'Show':
            # get and format the prediction
            tweet = str([values['_IN_']])
            print(tweet)
            p1 = model_NB.predict(tweet)
            p2 = model_SGD.predict(tweet)
            p3 = model_SVM.predict(tweet)
            result_p1 = sentiment_str(p1[0])
            result_p2 = sentiment_str(p2[0])
            result_p3 = sentiment_str(p3[0])
            print(result_p1)
            print(result_p2)
            print(result_p3)
            # change the "output" element to be the value of "input" element
            window['_TWEET_OUT_'].update(tweet)
            window['_NB_Output_'].update(result_p1)
            window['_SGD_Output_'].update(result_p2)
            window['_SVM_Output_'].update(result_p3)

    window.close()
