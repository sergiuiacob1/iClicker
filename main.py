import cv2
import time
from pynput.mouse import Listener
import logging
from app import App, AppOptions


def main():
    app = App()
    # option = int(input(app.getAppInstructions()))
    option = AppOptions.collectData
    if option == AppOptions.collectData:
        app.collectTrainingData()
        input('Press Enter when you are done')
        app.endDataCollection()
        app.displaySampleFromCollectedData()
    if option == AppOptions.predict:
        app.predictData()


if __name__ == '__main__':
    main()
