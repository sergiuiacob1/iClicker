# TODO delete what's not necessary
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
from ui.eye_contour import EyeContour
from ui.eye_widget import EyeWidget


class TrainingDataCollectorViewer(QtWidgets.QMainWindow):
    def __init__(self, webcam_capturer):
        super().__init__()
        self.webcam_capturer = webcam_capturer
        self.create_window()
        self.eye_widget = EyeWidget()

    def create_window(self):
        self.setWindowTitle('Data Collector')
        self.webcam_image_widget = QtWidgets.QLabel()
        self.left_eye_contour = EyeContour(self.webcam_image_widget)
        self.right_eye_contour = EyeContour(self.webcam_image_widget)
        # self.stop_button = build_button(
        #     'Stop', 'Stop Collecting Data', self.end_data_collection)
        # self.stop_button.setParent(self.webcam_image_widget)
        self.setCentralWidget(self.webcam_image_widget)

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
