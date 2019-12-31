from PyQt5 import QtWidgets
import random
import enum
import time
import json
import os
import numpy as np
import joblib
from train import train_model
from training_data_collecter import TrainingDataCollector
from webcam_capturer import WebcamCapturer
from data_viewer import DataViewer
from utils import get_screen_dimensions


class AppOptions(enum.Enum):
    collectData = 1
    predict = 2
    train_model = 3
    view_data = 4


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.webcamCapturer = WebcamCapturer()
        self.training_data_collector = TrainingDataCollector(
            self.webcamCapturer)
        self.data_viewer = DataViewer()

    def display_main_menu(self):
        self.setWindowTitle('iClicker')
        # self.resize(*get_screen_dimensions())
        self.resize(800, 600)
        # creating layout
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(QtWidgets.QVBoxLayout())
        self.top_menu_part = QtWidgets.QLabel('iClicker')
        self.bottom_menu_part = QtWidgets.QWidget()
        main_widget.layout().addWidget(self.top_menu_part)
        main_widget.layout().addWidget(self.bottom_menu_part)
        self.add_control_buttons()
        self.setCentralWidget(main_widget)
        self.show()

    def add_control_buttons(self):
        self.bottom_menu_part.setLayout(QtWidgets.QGridLayout())
        collect_data_button = QtWidgets.QPushButton('Collect data')
        collect_data_button.setToolTip('Collect training data')
        collect_data_button.clicked.connect(self.collect_training_data)

        train_button = QtWidgets.QPushButton('Train model')
        train_button.setToolTip('Train model based on collected data')
        train_button.clicked.connect(self.train_model)

        predict_button = QtWidgets.QPushButton('Predict')
        predict_button.setToolTip('Predict cursor position')
        predict_button.clicked.connect(self.predictData)

        view_data_button = QtWidgets.QPushButton('View data')
        view_data_button.setToolTip('View collected data')
        view_data_button.clicked.connect(self.view_data)

        buttons = [collect_data_button, train_button,
                   predict_button, view_data_button]
        for i in range(0, 2):
            for j in range(0, 2):
                self.bottom_menu_part.layout().addWidget(
                    buttons[i * 2 + j], i, j)

    def view_data(self):
        print('Getting collected data...')
        data = self.training_data_collector.getCollectedData()
        print(f'Displaying random photos from {len(data)} samples')
        self.data_viewer.view_data(data)

    def train_model(self):
        # first get data
        data = self.training_data_collector.getCollectedData()
        print(f'Loaded {len(data)} items')
        model, accuracy = train_model(data)
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

    def get_app_instructions(self):
        instructions = "The following options are available:\n"
        for option in AppOptions:
            instructions += repr(option) + "\n"
        instructions += "Enter your choice: "
        return instructions

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

    def collect_training_data(self):
        self.training_data_collector.start_collecting()
