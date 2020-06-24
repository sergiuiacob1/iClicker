import threading
import time
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

# My files
from src.webcam_capturer import get_webcam_image
from src.utils import setup_logger
import config as Config
# ui
from src.ui.eye_contour import EyeContour
from src.ui.eye_widget import EyeWidget
from src.ui.ui_utils import get_qimage_from_cv2, build_button
from src.ui.base_gui import BaseGUI

dc_logger = setup_logger('dc_logger', './logs/data_collector.log')


class DataCollectorGUI(BaseGUI):
    def __init__(self, controller):
        super().__init__(controller)
        self.eye_widget = EyeWidget()
        self.face_detector = None

    def create_window(self):
        self.setWindowTitle('Data Collector')
        self.webcam_image_widget = QtWidgets.QLabel()
        self.left_eye_contour = EyeContour(self.webcam_image_widget)
        self.right_eye_contour = EyeContour(self.webcam_image_widget)
        self.setCentralWidget(self.webcam_image_widget)

    def show_how_to_collect_data(self):
        widget = QtWidgets.QWidget()
        widget.setLayout(QtWidgets.QVBoxLayout())
        # text = QtWidgets.QLabel('Active collection:\nIn this type of collection, the mouse cursor will be moving on the screen and you have to follow it.\nYou can pause it using the SPACEBAR. You can increase or decrease it\'s speed using UP_ARROW and DOWN_ARROW.\n\nBackground collection:\nEvery time you click somewhere, a picture from the webcam will be taken and saved with the mouse cursor position.\nTo finish data collection, close the window.\n')
        text = QtWidgets.QLabel('Colectare activă:\nÎn acest mod de colectare, cursorul mouse-ului se va mișca pe ecran și va trebui să îl urmăriți cu ochii.\n Pentru a face o pauză, apăsați tasta SPACEBAR. Puteți crește/micșora viteza cursorului folosind tastele SĂGEATĂ SUS și SĂGEATĂ JOS.\n\nColectare pasivă:\nDe fiecare dată când apăsați click stânga undeva, o imagine capturată prin intermediul webcam-ului va fi salvată împreună cu poziția cursorului.\nPentru a termina colectarea de date, închideți fereastra ce se va deschide.\n')
        widget.layout().addWidget(text)
        active_button = build_button(
            'Colectare activă', 'Începere colectare activă', self.start_collecting, f_args=('active'))
        background_button = build_button(
            'Colectare pasivă', 'Începere colectare pasivă', self.start_collecting, f_args=('background'))

        widget.layout().addWidget(active_button)
        widget.layout().addWidget(background_button)
        self.choose_widget = widget
        widget.show()

    def start_collecting(self, collection_type):
        """Closes the widget that the user used to choose how the data should be collected and starts collecting data."""
        self.choose_widget.close()
        self.controller.start_collecting(collection_type)

    def start(self):
        # self.eye_widget.show()
        self.show()
        threading.Thread(target=self.show_webcam_images).start()

    def closeEvent(self, event):
        """This function is ran when the training data window is closed"""
        dc_logger.info('Closing DataCollectorGUI')
        # self.eye_widget.close()
        dc_logger.info('EyeWidget was closed')
        self.close()
        dc_logger.info('DataCollectorGUI closed')
        self.controller.end_data_collection()

    def keyPressEvent(self, e):
        super().keyPressEvent(e)
        if e.key() == Qt.Key_Up:
            self.controller.increase_speed()
        elif e.key() == Qt.Key_Down:
            self.controller.decrease_speed()
        elif e.key() == Qt.Key_Space:
            self.controller.pause()

    def show_webcam_images(self):
        """Target function for a thread showing images from webcam.

        Automatically stops when the training data window is closed
        """
        # Only do this as long as the window is visible
        dc_logger.info('Displaying images from webcam...')
        interval = 1/Config.UPDATE_FPS
        while self.isVisible():
            start = time.time()
            success, image = get_webcam_image()
            if success is False:
                continue
            # draw eye contours
            # threading.Thread(target=self.update_eye_contours,
            #                  args=(image,)).start()
            qt_image = get_qimage_from_cv2(image)
            self.webcam_image_widget.setPixmap(
                QtGui.QPixmap.fromImage(qt_image))
            end = time.time()
            if interval - (end - start) > 0:
                time.sleep(interval - (end - start))
        dc_logger.info('Stop displaying images from the webcam')

    def update_eye_contours(self, image):
        # TODO this isn't done
        return
        # contours = face_detector.get_eye_contours(image)
        # if len(contours) == 2:
        #     self.left_eye_contour.points = contours[0]
        #     self.right_eye_contour.points = contours[1]
        #     if (self.eye_widget.isVisible()):
        #         self.eye_widget.update(image, contours[0])
