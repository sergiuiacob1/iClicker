import random
import datetime
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from cv2 import cv2


widget_positions = {
    "image": 0,
}


class CircleOverlay(QtWidgets.QWidget):
    def __init__(self, center, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.center = center
        self.setFixedSize(parent.size())
        # self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        # self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 20))
        painter.drawEllipse(self.center[0], self.center[1], 20, 20)


class DataViewer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Viewer')
        self.setLayout(QtWidgets.QVBoxLayout())
        image_label = QtWidgets.QLabel()
        # placeholder for images
        self.layout().addWidget(image_label)
        # next button
        self.layout().addWidget(self.build_next_button())

        # initialise random so it doesn't give the same output
        random.seed(datetime.datetime.now())

    def view_data(self, data):
        self.data = data
        self.display_sample()

    def build_next_button(self):
        button = QtWidgets.QPushButton('Next')
        button.setToolTip('Display next sample')
        button.clicked.connect(lambda: self.display_sample())
        return button

    def display_sample(self):
        """Chooses a different item from the sample data and displays it to the user"""
        item = random.choice(self.data)
        image_label = QtWidgets.QLabel()
        sample_image = self.build_sample_image(item.image)
        image_label.setPixmap(QtGui.QPixmap.fromImage(sample_image))

        # Builds this widget as a child of image_label
        text_label = QtWidgets.QLabel(image_label)
        text_label.setText(
            f'{str(item.mousePosition)}\nVertical: {item.vertical}\nHorizontal: {item.horizontal}')

        # Builds this widget as a child of image_label
        center = self.get_scaled_mouse_position(item, (image_label.width(), image_label.height()))
        _ = CircleOverlay(center=center, parent=image_label)

        self.layout().replaceWidget(
            self.layout().itemAt(widget_positions["image"]).widget(), image_label)

        print(image_label.width())
        self.show()

    def get_scaled_mouse_position(self, item, size):
        width, height = size
        scaleW = item.mousePosition[0] / item.screenSize[0]
        scaleH = item.mousePosition[1] / item.screenSize[1]
        return (width * scaleW, height * scaleH)

    def build_sample_image(self, cv2_image):
        """Receives an image that was captured using OpenCV.

        Returns a `QImage`
        """
        rgb_array = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_array.shape
        bytesPerLine = ch * w
        return QtGui.QImage(rgb_array.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
