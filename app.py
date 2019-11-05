import enum
import json
import os
import numpy as np
import joblib
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from training_data_collecter import TrainingDataCollector
from webcam_capturer import WebcamCapturer


class AppOptions(enum.Enum):
    collectData = 1
    predict = 2
    trainModel = 3


class App:
    def __init__(self):
        self.webcamCapturer = WebcamCapturer()
        self.trainingDataCollector = TrainingDataCollector(self.webcamCapturer)

    def trainModel(self):
        # first get data
        data = self.trainingDataCollector.getCollectedData()
        print(f'Loaded {len(data)} items')
        X = np.array(list(map(lambda x: np.array(x.image).flatten(), data)))
        Y = np.array(list(map(lambda x: (x.horizontal, x.vertical), data)))
        print('Training neural network...')
        forest = RandomForestClassifier(
            n_estimators=100, random_state=1, verbose=1)
        multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
        predictions = multi_target_forest.fit(X, Y).predict(X)
        accuracy = sum([1 if x[0] == y[0] and x[1] == y[1] else 0 for x,
                        y in zip(Y, predictions)])/len(Y)
        print(f'Accuracy: {accuracy}')
        path = os.path.join(self._getModelsDirectoryPath(), 'model.pkl')
        print(f'Saving model in {path}')
        joblib.dump(multi_target_forest, path)

    def _getModelsDirectoryPath(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        path = config['modelsDirectoryPath']
        path = os.path.abspath(
            os.path.join(os.getcwd(), path))
        return path

    def getAppInstructions(self):
        instructions = "The following options are available:\n"
        for index, option in enumerate(AppOptions):
            instructions += str(index) + ': ' + str(option) + "\n"
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
        self.webcamCapturer.previewWebcam()
        

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
