import logging
import threading
import joblib
import json
import os
import time
import random
from typing import List
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# My files
from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer
from data_object import DataObject
import config as Config
import utils as Utils
# ui
from ui.data_collector_viewer import DataCollectorViewer


screenWidth, screenHeight = Utils.get_screen_dimensions()


class DataCollector(QtWidgets.QMainWindow):
    def __init__(self, webcam_capturer: WebcamCapturer):
        super().__init__()
        self.collected_data = []
        self.mouse_listener = MouseListener(self.on_mouse_click)
        self.webcam_capturer = webcam_capturer
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')
        self.collect_data_lock = threading.Lock()
        self.viewer = DataCollectorViewer(webcam_capturer)

    def closeEvent(self, event):
        """This function is ran when the training data window is closed"""
        self.end_data_collection()

    def start_collecting(self):
        print('Started collecting training data...')
        self.viewer.start()
        self.mouse_listener.start_listening()
        self.webcam_capturer.start_capturing()

    def save_collected_data(self):
        if len(self.collected_data) == 0:
            return
        self.collect_data_lock.acquire()

        try:
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
        self.viewer.stop()
        self.mouse_listener.stop_listening()
        self.webcam_capturer.stop_capturing()

    def on_mouse_click(self, x, y, button, pressed):
        if pressed:
            logging.info(f'{button} pressed at ({x}, {y})')
            success, webcamImage = self.webcam_capturer.get_webcam_image()
            if success is True:
                dataObject = DataObject(webcamImage, (x, y))
                self.collect_data_lock.acquire()
                self.collected_data.append(dataObject)
                self.collect_data_lock.release()

    def _get_data_directory_path(self):
        dataDirectoryPath = Config.data_directory_path
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
