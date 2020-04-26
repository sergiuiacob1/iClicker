import threading
import time
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt

# My files
from src.webcam_capturer import get_webcam_image
from src.utils import setup_logger
# ui
from src.ui.eye_contour import EyeContour
from src.ui.eye_widget import EyeWidget
from src.ui.ui_utils import get_qimage_from_cv2, build_button

dc_logger = setup_logger('dc_logger', './logs/data_collector.log')

class DataCollectorGUI(QtWidgets.QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.eye_widget = EyeWidget()
        self.create_window()
        self.face_detector = None


    def show_how_to_collect_data(self):
        widget = QtWidgets.QWidget()
        widget.setLayout(QtWidgets.QVBoxLayout())
        text = QtWidgets.QLabel('Active collection:\nIn this type of collection, the mouse cursor will be moving on the screen and you have to follow it.\nYou can pause it using the SPACEBAR. You can increase or decrease it\'s speed using UP_ARROW and DOWN_ARROW.\n\nBackground collection:\nEvery time you click somewhere, a picture from the webcam will be taken and saved with the mouse cursor position.\nTo finish data collection, close the window.\n')
        widget.layout().addWidget(text)
        active_button = build_button('Active collection', 'Start collecting data in active mode', self.start_collecting, f_args = ('active'))
        background_button = build_button('Background collection', 'Start collecting data in background mode', self.start_collecting, f_args = ('background'))

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

    # TODO derive this from BaseGUI and delete this below
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()
        elif e.key() == Qt.Key_Up:
            self.controller.increase_speed()
        elif e.key() == Qt.Key_Down:
            self.controller.decrease_speed()
        elif e.key() == Qt.Key_Space:
            self.controller.pause()

    def create_window(self):
        self.setWindowTitle('Data Collector')
        self.webcam_image_widget = QtWidgets.QLabel()
        self.left_eye_contour = EyeContour(self.webcam_image_widget)
        self.right_eye_contour = EyeContour(self.webcam_image_widget)
        self.setCentralWidget(self.webcam_image_widget)

    def show_webcam_images(self):
        """Target function for a thread showing images from webcam.

        Automatically stops when the training data window is closed
        """
        # Only do this as long as the window is visible
        dc_logger.info('Displaying images from webcam...')
        fps = 30
        while self.isVisible():
            success, image = get_webcam_image()
            if success is False:
                continue
            # draw eye contours
            # threading.Thread(target=self.update_eye_contours,
            #                  args=(image,)).start()
            qt_image = get_qimage_from_cv2(image)
            self.webcam_image_widget.setPixmap(
                QtGui.QPixmap.fromImage(qt_image))
            time.sleep(1.0/fps)
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
