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

from mouse_listener import MouseListener
from webcam_capturer import WebcamCapturer
from data_viewer import DataViewer
from face_detector import FaceDetector
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


class EyeWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Eye')
        self.setLayout(QtWidgets.QVBoxLayout())
        self.eye = QtWidgets.QLabel()
        self.layout().addWidget(self.eye)
        # TODO delete below
        self.resize(200, 200)

    def update(self, cv2_image, eye_contour):
        x_min = min([x[0] for x in eye_contour])
        x_max = max([x[0] for x in eye_contour])
        y_min = min([x[1] for x in eye_contour])
        y_max = max([x[1] for x in eye_contour])

        eye_portion = cv2_image[y_min:y_max, x_min:x_max]
        eye_portion = Utils.resize_cv2_image(
            eye_portion, fixed_dim=(Config.EYE_WIDTH, Config.EYE_HEIGHT))
        eye_portion = Utils.get_binary_thresholded_image(eye_portion)
        q_image = Utils.build_sample_image(eye_portion)
        self.eye.setPixmap(QtGui.QPixmap.fromImage(q_image))


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
    def __init__(self, webcam_capturer):
        super().__init__()
        self.collected_data = []
        self.mouse_listener = MouseListener(self.onMouseClick)
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
        self.mouse_listener.stop_listening()
        self.webcam_capturer.stop_capturing()

    def onMouseClick(self, x, y, button, pressed):
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
