import threading
import time
from PyQt5 import QtWidgets, QtGui

# My files
from webcam_capturer import WebcamCapturer
from face_detector import FaceDetector
# ui
from ui.eye_contour import EyeContour
from ui.eye_widget import EyeWidget
from ui.ui_utils import get_qimage_from_cv2


class DataCollectorViewer(QtWidgets.QMainWindow):
    def __init__(self, webcam_capturer: WebcamCapturer):
        super().__init__()
        self.webcam_capturer = webcam_capturer
        self.eye_widget = EyeWidget()
        self.create_window()

    def start(self):
        self.show()
        self.eye_widget.show()
        self.face_detector = FaceDetector()
        threading.Thread(target=self.show_webcam_images).start()

    def stop(self):
        ...

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
            qt_image = get_qimage_from_cv2(image)
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
