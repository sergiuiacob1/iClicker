from PyQt5 import QtWidgets
import threading
import random
import enum
import time
import json
import os
import numpy as np
import joblib
import sys

from train import train_model
from training_data_collecter import TrainingDataCollector, DataObject
from webcam_capturer import WebcamCapturer
from data_viewer import DataViewer
from utils import get_screen_dimensions, run_function_on_thread
from data_processing import process_data
import config as Config


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
        train_button.clicked.connect(
            lambda: run_function_on_thread(self.train_model))

        predict_button = QtWidgets.QPushButton('Predict')
        predict_button.setToolTip('Predict cursor position')
        predict_button.clicked.connect(
            lambda: run_function_on_thread(self.predict_data))

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
        # TODO put this on a thread?
        print('Getting collected data...')
        data = self.training_data_collector.get_collected_data()
        print(f'Displaying random photos from {len(data)} samples')
        self.data_viewer.view_data(data)

    def train_model(self):
        model = train_model()
        path = os.path.join(self._get_models_directory_path(), 'model.pkl')
        print(f'Saving model in {path}')
        # TODO remove joblib if not used
        model.save(path)
        # joblib.dump(model, path)

    def _get_models_directory_path(self):
        path = Config.models_directory_path
        path = os.path.abspath(
            os.path.join(os.getcwd(), path))
        os.makedirs(path, exist_ok=True)
        return path

    def predict_data(self):
        print('Loading trained model...')
        fps = 30
        model = self._get_trained_model()
        while True:
            time.sleep(1)
            success, image = self.webcamCapturer.getWebcamImage()
            if success is False:
                print('Failed capturing image')
                continue
            data = [DataObject(image, (0, 0))]
            data = process_data(data)
            if len(data) == 0:
                continue
            X = [(d[0][0] + d[0][1]).flatten() for d in data]
            X = np.array(X)
            prediction = model.predict(X)[0][0]
            if prediction > 0.5:
                print('LEFT')
            else:
                print('RIGHT')

    def _get_trained_model(self):
        path = self._get_models_directory_path()
        path = os.path.join(path, 'model.pkl')
        import keras
        model = keras.models.load_model(path)
        # TODO cleanup
        # model = joblib.load(os.path.join(path, 'model.pkl'))
        return model

    def collect_training_data(self):
        self.training_data_collector.start_collecting()
