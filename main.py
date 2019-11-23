import time
from pynput.mouse import Listener
import logging
from app import App, AppOptions


def main():
    app = App()
    option = int(input(app.getAppInstructions()))
    if option == AppOptions.collectData:
        app.collectTrainingData()
        input('Press Enter when you are done')
        app.endDataCollection()
        app.displaySampleFromCollectedData()
    if option == AppOptions.trainModel:
        app.trainModel()
    if option == AppOptions.predict:
        app.predictData()
        input('Press Enter when you are done')
    if option == AppOptions.viewData:
        app.viewData()


if __name__ == '__main__':
    main()
