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
import dlib
# TODO cleanup perhaps? shouldn't be using cv2 here
import cv2
from imutils import face_utils

from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer
from data_viewer import DataViewer
import utils as Utils
import config as Config


screenWidth, screenHeight = Utils.get_screen_dimensions()


class EyeContour(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.points = None
        self.setFixedSize(parent.size())

    def paintEvent(self, event):
        if self.points is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
        # sort points py x ascending, y descending (so they go counter clockwise if iterated)
        self.points = Utils.sort_points_counter_clockwise(self.points)
        for i in range(0, len(self.points) - 1):
            painter.drawLine(self.points[i][0], self.points[i]
                             [1], self.points[i + 1][0], self.points[i + 1][1])
        # complete the circle
        painter.drawLine(
            self.points[-1][0], self.points[-1][1], self.points[0][0], self.points[0][1])


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
        self.face_detector, self.face_predictor = None, None

    def closeEvent(self, event):
        """This function is ran when the training data window is closed"""
        self.end_data_collection()

    def create_window(self):
        self.setWindowTitle('Data Collector')
        self.webcam_image = QtWidgets.QLabel()
        self.left_eye_contour = EyeContour(self.webcam_image)
        self.right_eye_contour = EyeContour(self.webcam_image)
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
        if self.face_detector is None or self.face_predictor is None:
            threading.Thread(target=self.load_face_utils).start()

    def load_face_utils(self):
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(Config.face_landmarks_path)

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

            # draw eye contours
            threading.Thread(target=self.update_eye_contours,
                             args=(image,)).start()
            qt_image = Utils.build_sample_image(image)
            self.webcam_image.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            time.sleep(1/fps)
        print('Stop displaying images from the webcam')

    def update_eye_contours(self, image):
        if self.face_detector is None or self.face_predictor is None:
            return
        gray_image = Utils.convert_to_gray_image(image)
        rects = self.face_detector(gray_image, 0)
        if len(rects) > 0:
            shape = self.face_predictor(image, rects[0])
            shape = face_utils.shape_to_np(shape)
            (lStart,
                lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart,
                rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            self.left_eye_contour.points = shape[lStart:lEnd]
            self.right_eye_contour.points = shape[rStart:rEnd]

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
