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

    layout = [[sg.Text('Enter a possible tweet (280 chars or less):')],
              [sg.Input(key='_IN_', do_not_clear=False, size=(100, 5))],
              [sg.Button('Show'), sg.Button('Exit')],
              [sg.Text("Naives-Bayes Classifier")],
              [sg.Text("The sentiment of your tweet is: ")],
              [sg.Text(size=(100, 5), key='_NB_Output_')]]

    window = sg.Window('Tweet Sentiment Analysis', layout)

    while True:  # Event Loop
        event, values = window.read()  # can also be written as event, values = window()
        print(event, values)
        if event is None or event == 'Exit':
            break
        if event == 'Show':
            # get and format the prediction
            p1 = model_NB.predict([values['_IN_']])

            result = "The sentence: \n'{}' \nhas a {} sentiment".format(str(values['_IN_']), sentiment_str(p1[0]))
            print(result)
            # change the "output" element to be the value of "input" element
            window['_NB_Output_'].update(result)

    window.close()
