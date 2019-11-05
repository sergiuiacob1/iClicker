import cv2
import time
from pynput.mouse import Listener
import logging
from app import App, AppOptions


def main():
    app = App()
    # option = int(input(app.getAppInstructions()))
    option = AppOptions.predict
    if option == AppOptions.collectData:
        app.collectTrainingData()
        input('Press Enter when you are done')
        app.endDataCollection()
        app.displaySampleFromCollectedData()
    if option == AppOptions.trainModel:
        app.trainModel()
    if option == AppOptions.predict:
        app.predictData()
        input ('Press Enter when you are done')


if __name__ == '__main__':
    main()
