import random
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from cv2 import cv2


class DataViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Viewer')
        self.main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.main_layout)
        self.main_window = QtWidgets.QMainWindow()
        self.next_button = self.build_next_button()
        self.main_layout.addWidget(self.main_window)
        self.main_layout.addWidget(QtWidgets.QLabel()) # placeholder for images
        self.main_layout.addWidget(self.next_button)

    def view_data(self, data):
        self.data = data
        # self.main_layout.addWidget(self.main_window)
        # self.main_layout.addWidget(self.next_button)
        # self.show()
        self.display_sample()

    def build_next_button(self):
        button = QtWidgets.QPushButton('Next')
        button.setToolTip('Display next sample')
        button.clicked.connect(lambda: self.display_sample())
        return button

    @QtCore.pyqtSlot()
    def on_click(self):
        self.display_sample

    def display_sample(self):
        item = random.choice(self.data)
        label = QtWidgets.QLabel()
        sample_image = self.build_sample_image(item.image)
        label.setPixmap(QtGui.QPixmap.fromImage(sample_image))

        self.main_layout.replaceWidget(self.main_layout.itemAt(1).widget(), label)
        self.show()

    def build_sample_image(self, cv2_image):
        """Receives an image that was captured using OpenCV.

        Returns a `QImage`
        """
        rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_array.shape
        bytesPerLine = ch * w
        return QtGui.QImage(rgb_array.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
