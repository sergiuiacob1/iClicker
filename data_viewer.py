import random
import datetime
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui
from cv2 import cv2
from ui.ui_utils import get_qimage_from_cv2
import seaborn as sns
import matplotlib.pyplot as plt

widget_positions = {
    "image": 0,
}


class CircleOverlay(QtWidgets.QWidget):
    def __init__(self, center, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.center = center
        self.setFixedSize(parent.size())

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.red, 20))
        painter.drawEllipse(self.center[0], self.center[1], 20, 20)


class StatisticsViewer():
    def show_statistics(self, data):
        x = [x.mousePosition[0] for x in data]
        y = [x.mousePosition[1] for x in data]
        ax = sns.jointplot(x=x, y=y)
        plt.show()


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

        self.statistics_viewer = StatisticsViewer()

        # initialise random so it doesn't give the same output
        random.seed(datetime.datetime.now())

    def view_data(self, data):
        if len(data) == 0:
            print("No data to view")
            return
        self.data = data
        self.statistics_viewer.show_statistics(data)
        self.display_sample()
        # focus the window
        self.raise_()

    def build_next_button(self):
        button = QtWidgets.QPushButton('Next')
        button.setToolTip('Display next sample')
        button.clicked.connect(lambda: self.display_sample())
        return button

    def display_sample(self):
        """Chooses a different item from the sample data and displays it to the user"""
        item = random.choice(self.data)
        image_label = QtWidgets.QLabel()
        sample_image = get_qimage_from_cv2(item.image)
        image_label.setPixmap(QtGui.QPixmap.fromImage(sample_image))

        # Builds this widget as a child of image_label
        text_label = QtWidgets.QLabel(image_label)
        text_label.setText(
            f'{str(item.mousePosition)}\nVertical: {item.vertical}\nHorizontal: {item.horizontal}')

        # Builds this widget as a child of image_label
        center = self.get_scaled_mouse_position(
            item, (image_label.width(), image_label.height()))
        _ = CircleOverlay(center=center, parent=image_label)

        self.layout().replaceWidget(self.layout().itemAt(
            widget_positions["image"]).widget(), image_label)

        self.show()

    # TODO this is broken
    def get_scaled_mouse_position(self, item, size):
        width, height = size
        scaleW = item.mousePosition[0] / item.screenSize[0]
        scaleH = item.mousePosition[1] / item.screenSize[1]
        return (width * scaleW, height * scaleH)
