import random
import enum
import time
import json
import os
import numpy as np
import joblib
from train import trainModel
from training_data_collecter import TrainingDataCollector
from webcam_capturer import WebcamCapturer
from data_viewer import DataViewer


class AppOptions(enum.Enum):
    collectData = 1
    predict = 2
    trainModel = 3
    viewData = 4


class App():
    def __init__(self):
        self.webcamCapturer = WebcamCapturer()
        self.trainingDataCollector = TrainingDataCollector(self.webcamCapturer)
        self.data_viewer = DataViewer()

    def viewData(self):
        print('Getting collected data...')
        data = self.trainingDataCollector.getCollectedData()
        print(f'Displaying random photos from {len(data)} samples')
        self.data_viewer.view_data(data)

    def trainModel(self):
        # first get data
        data = self.trainingDataCollector.getCollectedData()
        print(f'Loaded {len(data)} items')
        model, accuracy = trainModel(data)
        print(f'Accuracy: {accuracy}')
        path = os.path.join(self._getModelsDirectoryPath(), 'model.pkl')
        print(f'Saving model in {path}')
        joblib.dump(model, path)

    def _getModelsDirectoryPath(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        path = config['modelsDirectoryPath']
        path = os.path.abspath(
            os.path.join(os.getcwd(), path))
        return path

    def getAppInstructions(self):
        instructions = "The following options are available:\n"
        for option in AppOptions:
            instructions += repr(option) + "\n"
        instructions += "Enter your choice: "
        return instructions

    # def processUserOption(self, option):
    #     print (f'Got option: {option}')
    #     if option == AppOptions.collectData:
    #         self.collectTrainingData()
    #     if option == AppOptions.predict:
    #         self.predictData()

    def predictData(self):
        print('Loading trained model...')
        model = self._getTrainedModel()
        # self.webcamCapturer.previewWebcam()
        while True:
            success, image = self.webcamCapturer.getWebcamImage()
            if success is False:
                print('Failed capturing image')
                continue
            X = np.array(image).flatten()
            X = X.reshape(1, -1)
            print(model.predict(X))
            time.sleep(1)

    def _getTrainedModel(self):
        path = self._getModelsDirectoryPath()
        model = joblib.load(os.path.join(path, 'model.pkl'))
        return model

    def collectTrainingData(self):
        self.trainingDataCollector.startCollecting()

    def endDataCollection(self):
        self.trainingDataCollector.endDataCollection()

    def displaySampleFromCollectedData(self):
        self.trainingDataCollector.displaySampleFromCollectedData()


if __name__ == '__main__':
    app = App()
