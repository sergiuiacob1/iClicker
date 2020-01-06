from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer
from data_viewer import DataViewer
from utils import get_screen_dimensions, build_button, build_sample_image
import logging
import threading
import joblib
import json
import os
import time
import random
from typing import List
import numpy as np
from PyQt5 import QtWidgets
from PyQt5 import QtGui


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


class TrainingDataCollector(QtWidgets.QMainWindow):
    def __init__(self, webcamCapturer):
        super().__init__()
        self.collected_data = []
        self.mouseListener = MouseListener(self.onMouseClick)
        self.webcamCapturer = webcamCapturer
        self.create_window()
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')
        self.collect_data_lock = threading.Lock()

    def closeEvent(self, event):
        """This function is ran when the training data window is closed"""
        self.end_data_collection()

    def create_window(self):
        self.setWindowTitle('Data Collector')
        self.webcam_image = QtWidgets.QLabel()
        # self.stop_button = build_button(
        #     'Stop', 'Stop Collecting Data', self.end_data_collection)
        # self.stop_button.setParent(self.webcam_image)
        self.setCentralWidget(self.webcam_image)

    def start_collecting(self):
        print('Started collecting training data...')
        self.show()
        threading.Thread(target=self.show_webcam_images).start()
        self.mouseListener.startListening()
        self.webcamCapturer.startCapturing()

    def show_webcam_images(self):
        """Target function for a thread showing images from webcam.

        Automatically stops when the training data window is closed
        """
        # Only do this as long as the window is visible
        print('Displaying images from webcam...')
        fps = 30
        while self.isVisible():
            success, image = self.webcamCapturer.getWebcamImage(
                start_if_not_started=False)
            if success is False:
                continue
            qt_image = build_sample_image(image)
            self.webcam_image.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            time.sleep(1/fps)
        print('Stop displaying images from the webcam')

    def save_collected_data(self):
        self.collect_data_lock.acquire()
        try:
            if len(self.collected_data) > 0:
                secondsSinceEpoch = time.time()
                path = os.path.join(
                    self._get_data_directory_path(), f'{secondsSinceEpoch}.pkl')
                print(f'Saving {len(self.collected_data)} samples in {path}')
                joblib.dump(self.collected_data, path)
                self.collected_data = []
        except Exception as e:
            print(f'Could not save data: {e}')
        self.collect_data_lock.release()

    def end_data_collection(self):
        threading.Thread(target=self.save_collected_data).start()
        # close the Qt window
        self.close()
        self.mouseListener.stopListening()
        self.webcamCapturer.stopCapturing()

    def onMouseClick(self, x, y, button, pressed):
        if pressed:
            logging.info(f'{button} pressed at ({x}, {y})')
            success, webcamImage = self.webcamCapturer.getWebcamImage()
            if success is True:
                dataObject = DataObject(webcamImage, (x, y))
                self.collect_data_lock.acquire()
                self.collected_data.append(dataObject)
                self.collect_data_lock.release()

    def displaySampleFromCollectedData(self):
        sample = random.choice(self.collected_data)
        print(sample)

    def _get_data_directory_path(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        dataDirectoryPath = config['dataDirectoryPath']
        dataDirectoryPath = os.path.abspath(
            os.path.join(os.getcwd(), dataDirectoryPath))
        os.makedirs(dataDirectoryPath, exist_ok=True)
        return dataDirectoryPath

    def get_collected_data(self) -> List[DataObject]:
        # TODO save somewhere how many items I have in order to avoid list appendings
        dataPath = self._get_data_directory_path()
        data = []
        for r, _, f in os.walk(dataPath):
            for file in f:
                if file.endswith('.pkl'):
                    currentData = joblib.load(os.path.join(r, file))
                    data.extend(currentData)
        return data
