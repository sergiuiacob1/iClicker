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
from data_viewer import DataViewer
from face_detector import FaceDetector
from data_object import DataObject
import utils as Utils
import config as Config
# ui
# TODO delete these cuz they aren't necessary
from ui.eye_contour import EyeContour
from ui.eye_widget import EyeWidget
from ui.training_data_collector_viewer import TrainingDataCollectorViewer


screenWidth, screenHeight = Utils.get_screen_dimensions()


class TrainingDataCollector(QtWidgets.QMainWindow):
    def __init__(self, webcam_capturer):
        super().__init__()
        self.collected_data = []
        self.mouse_listener = MouseListener(self.on_mouse_click)
        self.webcam_capturer = webcam_capturer
        self.create_window()
        self.eye_widget = EyeWidget()
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')
        self.collect_data_lock = threading.Lock()

    def closeEvent(self, event):
        """This function is ran when the training data window is closed"""
        self.end_data_collection()

    def create_window(self):
        self.setWindowTitle('Data Collector')
        self.webcam_image_widget = QtWidgets.QLabel()
        self.left_eye_contour = EyeContour(self.webcam_image_widget)
        self.right_eye_contour = EyeContour(self.webcam_image_widget)
        # self.stop_button = build_button(
        #     'Stop', 'Stop Collecting Data', self.end_data_collection)
        # self.stop_button.setParent(self.webcam_image_widget)
        self.setCentralWidget(self.webcam_image_widget)

    def start_collecting(self):
        print('Started collecting training data...')
        self.show()
        self.eye_widget.show()
        self.mouse_listener.start_listening()
        self.webcam_capturer.start_capturing()
        self.face_detector = FaceDetector()
        threading.Thread(target=self.show_webcam_images).start()

    def show_webcam_images(self):
        """Target function for a thread showing images from webcam.

        Automatically stops when the training data window is closed
        """
        # Only do this as long as the window is visible
        print('Displaying images from webcam...')
        fps = 30
        while self.isVisible():
            success, image = self.webcam_capturer.get_webcam_image(
                start_if_not_started=False)
            if success is False:
                continue

            # draw eye contours
            threading.Thread(target=self.update_eye_contours,
                             args=(image,)).start()
            qt_image = Utils.build_sample_image(image)
            self.webcam_image_widget.setPixmap(
                QtGui.QPixmap.fromImage(qt_image))
            time.sleep(1/fps)
        print('Stop displaying images from the webcam')

    def update_eye_contours(self, image):
        contours = self.face_detector.get_eye_contours(image)
        if len(contours) == 2:
            self.left_eye_contour.points = contours[0]
            self.right_eye_contour.points = contours[1]
            self.eye_widget.update(image, contours[0])

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
        self.close()
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
