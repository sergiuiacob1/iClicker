from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtCore import Qt

# My files
from src.utils import get_screen_dimensions
from src.ui.base_gui import BaseGUI
import config as Config

screen_width, screen_height = get_screen_dimensions()


class PredictorGUI(BaseGUI):
    def __init__(self, controller):
        super().__init__(controller)
        self.prediction = None
        self.mouth_is_opened = None
        self.coordinates = {}
        dx = screen_width / Config.grid_size
        dy = screen_height / Config.grid_size
        for i in range(0, Config.grid_size):
            for j in range(0, Config.grid_size):
                cell = i * Config.grid_size + j
                self.coordinates[cell] = (
                    j * dx, i * dy, (j + 1) * dx, (i + 1) * dy)

    def create_window(self):
        self.setWindowTitle('Predictor')
        self.resize(screen_width, screen_height)

    def update_prediction(self, prediction):
        self.prediction = prediction
        # redraw the widget
        self.update()

    def update_info (self, info):
        self.info = info
        self.update()

    def paintCells(self):
        painter = QPainter()
        painter.begin(self)
        painter.setBrush(QBrush(Qt.red, Qt.DiagCrossPattern))
        painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        font = QFont()
        font.setPointSize(font.pointSize() * 4)
        painter.setFont(font)
        for i in range(0, len(self.coordinates)):
            painter.drawRect(*(self.coordinates[i]))
            center_x = (self.coordinates[i][0] + self.coordinates[i][2])/2
            center_y = (self.coordinates[i][1] + self.coordinates[i][3])/2
            painter.drawText(center_x, center_y, f'{i}')
        painter.end()

    def paintEvent(self, event):
        self.paintCells()
        if self.prediction is None:
            return

        painter = QPainter()
        painter.begin(self)
        painter.setBrush(QBrush(Qt.green, Qt.DiagCrossPattern))
        painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
        c = self.coordinates[self.prediction]
        # i need width and height for the last two, but I currently have the coordinates for the bottom right point
        painter.drawRect(c[0], c[1], c[2] - c[0], c[3] - c[1])

        # draw info about the mouth
        self._draw_mouth_info(event, painter)
        painter.end()

    def _draw_mouth_info(self, event, qp):
        # qp.setPen(QColor(168, 34, 3))
        # qp.drawText(event.rect(), Qt.AlignTop, f'Mouth is opened: {self.info["mouth_is_opened"]}\nLeft eye ratio: {int(self.info["eyes"][1][0]*100)/100}')
        qp.drawText(event.rect(), Qt.AlignTop, f'Mouth is opened: {self.info["mouth_is_opened"]}\nEyes open: {self.info["eyes"]}')