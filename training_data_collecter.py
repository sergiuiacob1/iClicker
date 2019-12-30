from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer
from data_viewer import DataViewer
from utils import get_screen_dimensions
import logging
import joblib
import json
import os
import time
import random
from typing import List
import numpy as np


screenWidth, screenHeight = get_screen_dimensions()


class DataObject:
    def __init__(self, image, mousePosition):
        self.image = image
        self.mousePosition = mousePosition
        self.screenSize = (screenWidth, screenHeight)
        # x is for width, y is for height
        # (x, y) = mousePosition
        self.horizontal = 0 if self.mousePosition[0] < self.screenSize[0]/2 else 1
        self.vertical = 0 if self.mousePosition[1] < self.screenSize[1]/2 else 1

    def __str__(self):
        return f'Image Size: {len(self.image)}, mouse position: {self.mousePosition}, screen size: {self.screenSize}, (vertical, horizontal): {self.vertical, self.horizontal}'


class TrainingDataCollector:
    def __init__(self, webcamCapturer):
        self.collectedData = []
        self.mouseListener = MouseListener(self.onMouseClick)
        self.webcamCapturer = webcamCapturer
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')

    def startCollecting(self):
        print('Started collecting training data...')
        self.mouseListener.startListening()
        self.webcamCapturer.startCapturing()

    def endDataCollection(self):
        secondsSinceEpoch = time.time()
        dataDirectoryPath = os.path.join(
            self._getDataDirectoryPath(), f'{secondsSinceEpoch}.pkl')
        print(f'Saving collected data in {dataDirectoryPath}')
        joblib.dump(self.collectedData, dataDirectoryPath)

    def onMouseClick(self, x, y, button, pressed):
        if pressed:
            logging.info(f'{button} pressed at ({x}, {y})')
            success, webcamImage = self.webcamCapturer.getWebcamImage()
            if success is True:
                dataObject = DataObject(webcamImage, (x, y))
                self.collectedData.append(dataObject)

    def displaySampleFromCollectedData(self):
        sample = random.choice(self.collectedData)
        print(sample)

    def _getDataDirectoryPath(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        dataDirectoryPath = config['dataDirectoryPath']
        dataDirectoryPath = os.path.abspath(
            os.path.join(os.getcwd(), dataDirectoryPath))
        return dataDirectoryPath

    def getCollectedData(self) -> List[DataObject]:
        # TODO save somewhere how many items I have in order to avoid list appendings
        dataPath = self._getDataDirectoryPath()
        data = []
        for r, _, f in os.walk(dataPath):
            for file in f:
                if file.endswith('.pkl'):
                    currentData = joblib.load(os.path.join(r, file))
                    data.extend(currentData)
        return data
