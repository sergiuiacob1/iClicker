from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer
import logging
import tkinter as tk
import joblib
import json
import os
import time
import random


def getScreenDimensions():
    root = tk.Tk()
    return root.winfo_screenwidth(), root.winfo_screenheight()


screenWidth, screenHeight = getScreenDimensions()


class DataObject:
    def __init__(self, image, mousePosition):
        self.image = image
        self.mousePosition = mousePosition
        self.isCorrect = True
        self.screenSize = (screenWidth, screenHeight)
        # x is for width, y is for height
        # (x, y) = mousePosition
        self.horizontal = 0 if self.mousePosition[0] < self.screenSize[0]/2 else 1
        self.vertical = 0 if self.mousePosition[1] < self.screenSize[1]/2 else 1

    def __str__(self):
        return f'Image Size: {len(self.image)}, mouse position: {self.mousePosition}, screen size: {self.screenSize}, (vertical, horizontal): {self.vertical, self.horizontal}'


class TrainingDataCollector:
    def __init__(self):
        self.collectedData = []
        self.mouseListener = MouseListener(self.onMouseClick)
        self.webcamCapturer = WebcamCapturer()
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')

    def startCollecting(self):
        print('Started collecting training data...')
        self.mouseListener.startListening()
        self.webcamCapturer.startCapturing()

    def endDataCollection(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        dataDirectoryPath = config['dataDirectoryPath']
        secondsSinceEpoch = time.time()
        dataDirectoryPath = os.path.abspath(os.path.join(
            os.getcwd(), dataDirectoryPath, f'{secondsSinceEpoch}.pkl'))
        print(f'Saving collected data in {dataDirectoryPath}')
        joblib.dump(self.collectedData, dataDirectoryPath)

    def onMouseClick(self, x, y, button, pressed):
        if pressed:
            logging.info(f'{button} pressed at ({x}, {y})')
            success, webcamImage = self.webcamCapturer.getWebcamImage()
            if success:
                dataObject = DataObject(webcamImage, (x, y))
                self.collectedData.append(dataObject)
                # self.webcamCapturer.saveCurrentWebcamImage('data')

    def displaySampleFromCollectedData(self):
        sample = random.choice(self.collectedData)
        print(sample)
